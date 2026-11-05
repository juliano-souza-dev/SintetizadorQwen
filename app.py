import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_DISABLE_XET"] = "1"

import re
import torch
import soundfile as sf
import librosa
from pathlib import Path
from qwen_tts import Qwen3TTSModel
import gradio as gr
import json
import shutil
import traceback
from datetime import datetime

from tts_service import TTSService, registrar_log
from job_queue import JobQueue

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
VOICES_DIR = DATA_DIR / "voices"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

VOICES_FILE = DATA_DIR / "voices.json"
HISTORY_FILE = DATA_DIR / "history.json"
CAMINHO_MODELO = r"C:\Users\jsouz\Projects\nova-generator\models\qwen3-tts\Qwen3-TTS-12Hz-1.7B-Base"

tts_engine = TTSService(CAMINHO_MODELO)

def processar_job_sintese(payload: dict) -> str:
    """Executa a síntese usando a instância única do Qwen3-TTS."""
    nome_voz = payload["nome_voz"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    arquivo_saida = str(OUTPUTS_DIR / f"audio_{timestamp}.wav")

    if payload["usar_chunking"] and "[" in payload["texto"] and "]" in payload["texto"]:
        chunks = parse_texto_para_chunks(payload["texto"])
        tts_engine.synthesize_chunks(
            ref_audio_path=payload["audio_path"],
            ref_text=payload["transcricao"],
            chunks=chunks,
            output_path=arquivo_saida,
            temperature=payload["temperatura"]
        )
    else:
        texto_limpo = payload["texto"].replace("[", "").replace("]", "")
        tts_engine.synthesize(
            ref_audio_path=payload["audio_path"],
            ref_text=payload["transcricao"],
            text_to_speak=texto_limpo,
            output_path=arquivo_saida,
            temperature=payload["temperatura"]
        )

    historico = carregar_historico()
    historico.insert(0, {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "voz": nome_voz,
        "texto": payload["texto"],
        "arquivo": arquivo_saida
    })
    salvar_json(HISTORY_FILE, historico)
    return arquivo_saida


# Um único Worker protege a instância do modelo contra chamadas concorrentes.
fila_jobs = JobQueue(processar_job_sintese, max_history=100)

def carregar_json(arquivo, padrao):
    if os.path.exists(arquivo):
        try:
            with open(arquivo, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            registrar_log(f"[ERRO JSON] {arquivo}: {e}")
    return padrao

def salvar_json(arquivo, dados):
    try:
        with open(arquivo, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=4)
    except Exception as e:
        registrar_log(f"[ERRO JSON] {arquivo}: {e}")

def carregar_biblioteca():
    return carregar_json(VOICES_FILE, {})

def carregar_historico():
    return carregar_json(HISTORY_FILE, [])

def obter_total_historico():
    return len(carregar_historico())

def atualizar_tabela_historico(busca_texto="", filtro_voz="Todas"):
    historico = carregar_historico()
    linhas = []
    for h in historico:
        voz = h.get("voz", "")
        texto = h.get("texto", "")
        data_hora = h.get("data", "")
        arquivo = h.get("arquivo", "")

        if filtro_voz != "Todas" and voz != filtro_voz:
            continue
        if busca_texto and busca_texto.lower() not in texto.lower():
            continue

        linhas.append([data_hora, voz, texto, arquivo])
    return linhas

def renderizar_cards_vozes(busca=""):
    biblioteca = carregar_biblioteca()
    if not biblioteca:
        return "<div class='empty-card'>Nenhuma voz cadastrada.</div>"
    
    html = "<div class='voice-cards-grid'>"
    for nome, dados in biblioteca.items():
        if busca and busca.lower() not in nome.lower():
            continue
        iniciais = "".join([palavra[0].upper() for palavra in nome.split()[:2]])
        html += f"""
        <div class='voice-card'>
            <div class='voice-avatar'>{iniciais}</div>
            <div class='voice-info'>
                <div class='voice-name'>{nome}</div>
                <div class='voice-lang'>PT-BR / EN · Referência Ativa</div>
            </div>
        </div>
        """
    html += "</div>"
    return html

def parse_texto_para_chunks(texto: str) -> list:
    """
    Se o utilizador marcar blocos em inglês entre colchetes [como este],
    cria automaticamente chunks dedicados com as pausas configuradas.
    """
    partes = re.split(r"(\[[^\]]+\])", texto)
    chunks = []
    for parte in partes:
        p = parte.strip()
        if not p:
            continue
        if p.startswith("[") and p.endswith("]"):
            chunks.append({
                "text": p[1:-1].strip(),
                "lang": "en",
                "pause_after_ms": 180
            })
        else:
            chunks.append({
                "text": p,
                "lang": "pt",
                "pause_after_ms": 150
            })
    return chunks

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, sans-serif !important; }
body, .gradio-container { background-color: #0F1115 !important; color: #E5E7EB !important; }

.app-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 20px; background-color: #171A21;
    border: 1px solid #262B36; border-radius: 12px; margin-bottom: 20px;
}
.header-left { display: flex; align-items: center; gap: 14px; }
.header-icon {
    width: 44px; height: 44px; background: rgba(249, 115, 22, 0.15);
    border: 1px solid rgba(249, 115, 22, 0.3); border-radius: 10px;
    display: flex; align-items: center; justify-content: center; font-size: 22px;
}
.header-titles h1 { font-size: 20px; font-weight: 700; color: #FFF; margin: 0; }
.header-titles p { font-size: 13px; color: #9CA3AF; margin: 0; }
.status-badge {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
    border-radius: 20px; background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3); color: #22C55E; font-size: 13px; font-weight: 500;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background-color: #22C55E; }

.tab-nav {
    background-color: #171A21 !important; border: 1px solid #262B36 !important;
    border-radius: 10px !important; padding: 4px !important; gap: 6px !important; display: flex !important;
}
.tab-nav button {
    background: transparent !important; border: none !important; color: #9CA3AF !important;
    border-radius: 8px !important; padding: 8px 16px !important; font-size: 14px !important;
    min-height: 44px !important;
}
.tab-nav button.selected { background-color: #F97316 !important; color: #FFF !important; font-weight: 600 !important; }

.card-panel { background-color: #171A21 !important; border: 1px solid #262B36 !important; border-radius: 12px !important; padding: 16px !important; }
textarea, input[type="text"] { background-color: #0F1115 !important; border: 1px solid #262B36 !important; color: #E5E7EB !important; border-radius: 8px !important; }
textarea:focus, input[type="text"]:focus { border-color: #F97316 !important; }

.btn-primary { background-color: #F97316 !important; border: none !important; color: #FFF !important; font-weight: 600 !important; border-radius: 8px !important; min-height: 44px !important; }
.btn-primary:hover { background-color: #EA580C !important; }

.voice-cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 8px; max-height: 200px; overflow-y: auto; }
.voice-card { display: flex; align-items: center; gap: 10px; padding: 10px; background-color: #0F1115; border: 1px solid #262B36; border-radius: 10px; }
.voice-avatar { width: 36px; height: 36px; border-radius: 50%; background: #262B36; color: #F97316; display: flex; align-items: center; justify-content: center; font-weight: 700; }
.voice-name { font-size: 14px; font-weight: 600; color: #FFF; }
.voice-lang { font-size: 11px; color: #9CA3AF; }

.toast-banner { padding: 12px 16px; border-radius: 8px; font-size: 14px; font-weight: 500; margin-bottom: 12px; }
.toast-success { background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.3); color: #22C55E; }
.toast-error { background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); color: #EF4444; }

.app-footer { display: flex; justify-content: space-between; align-items: center; margin-top: 32px; padding-top: 16px; border-top: 1px solid #262B36; font-size: 12px; color: #6B7280; }
"""

with gr.Blocks(title="Qwen3-TTS · Síntese de Voz", css=CUSTOM_CSS) as demo:
    
    gr.HTML("""
    <div class='app-header'>
        <div class='header-left'>
            <div class='header-icon'>🎙️</div>
            <div class='header-titles'>
                <h1>Qwen3-TTS</h1>
                <p>Síntese de voz didática · Code-Switching PT-BR / EN</p>
            </div>
        </div>
        <div class='status-badge'>
            <div class='status-dot'></div>
            <span>Pronto</span>
        </div>
    </div>
    """)

    total_hist = obter_total_historico()
    with gr.Tabs():
        
        with gr.TabItem("🎧 Sintetizar Nova Frase"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="card-panel"):
                        gr.Markdown("### 1. Selecionar Voz do Narrador")
                        biblioteca = carregar_biblioteca()
                        nomes_iniciais = list(biblioteca.keys())
                        
                        dropdown_voz = gr.Dropdown(
                            choices=nomes_iniciais,
                            value=nomes_iniciais[0] if nomes_iniciais else None,
                            label="Voz Ativa"
                        )
                        cards_html = gr.HTML(value=renderizar_cards_vozes())

                    with gr.Group(elem_classes="card-panel"):
                        gr.Markdown("### 2. Texto para Locução Didática")
                        
                        input_texto_narrador = gr.Textbox(
                            label="",
                            placeholder="Dica: Use colchetes para destacar inglês nativo no modo chunking, ex.: Por isso dizemos... [I am watching a series].",
                            lines=5,
                            value="Por isso, a gente diz... [I am watching a series]. Ou então... [Let's watch the game!] A imagem se mexe, e você acompanha."
                        )
                        
                        with gr.Row():
                            chk_chunking = gr.Checkbox(
                                label="Modo Alta Fidelidade (Chunking por Idioma com Concatenação)",
                                value=False,
                                info="Divide frases em PT e EN isolados com micro-pausas. Mais natural na pronúncia, porém mais lento na CPU."
                            )
                            slider_temp = gr.Slider(minimum=0.5, maximum=0.8, value=0.65, step=0.05, label="Temperature (0.65 ideal)")

                        btn_sintetizar = gr.Button("⚡ Gerar Áudio", elem_classes="btn-primary")

                with gr.Column(scale=4):
                    toast_status = gr.HTML("<div class='toast-banner toast-success'>🟢 Sistema pronto para síntese</div>")
                    
                    with gr.Group(elem_classes="card-panel"):
                        gr.Markdown("### Resultado da Locução")
                        audio_output = gr.Audio(label="", interactive=False)

                    gr.Markdown("### Fila de Processamento")
                    tabela_jobs = gr.Dataframe(
                        headers=["Job", "Estado", "Criado em", "Iniciado em", "Finalizado em", "Erro"],
                        value=[],
                        interactive=False,
                        wrap=True
                    )

                    with gr.Accordion("Detalhes da Voz de Referência", open=False):
                        preview_audio_ref = gr.Audio(label="Áudio de Origem", interactive=False)
                        preview_texto_ref = gr.Textbox(label="Transcrição da Referência", interactive=False, lines=2)

        with gr.TabItem("➕ Cadastrar Nova Voz"):
            with gr.Group(elem_classes="card-panel"):
                gr.Markdown("### Cadastro de Nova Matriz de Voz (3 a 5 segundos)")
                with gr.Row():
                    with gr.Column():
                        novo_audio_ref = gr.Audio(type="filepath", label="Áudio de Referência (WAV limpo)", sources=["upload", "microphone"])
                    with gr.Column():
                        nova_transcricao = gr.Textbox(label="Transcrição Exata", lines=2)
                        novo_nome_voz = gr.Textbox(label="Nome do Narrador")
                        btn_salvar_voz = gr.Button("Salvar na Biblioteca", elem_classes="btn-primary")
                        status_salvamento = gr.HTML("")

        with gr.TabItem(f"📜 Histórico ({total_hist})"):
            with gr.Group(elem_classes="card-panel"):
                with gr.Row():
                    filtro_busca_hist = gr.Textbox(label="Buscar texto", scale=4)
                    filtro_voz_hist = gr.Dropdown(choices=["Todas"] + nomes_iniciais, value="Todas", label="Filtrar voz", scale=2)

                tabela_historico = gr.Dataframe(
                    headers=["Data/Hora", "Voz", "Texto", "Arquivo Salvo"],
                    value=atualizar_tabela_historico(),
                    interactive=False,
                    wrap=True
                )
                with gr.Row():
                    caminho_selecionado = gr.Textbox(label="Arquivo Selecionado", interactive=False)
                    player_historico = gr.Audio(label="Reproduzir", interactive=False)

    gr.HTML("""
    <div class='app-footer'>
        <div>Qwen3-TTS Local Engine · CPU (bfloat16)</div>
        <div>Pronto para Reels e Aulas Didáticas</div>
    </div>
    """)

    def sync_dados_voz(nome_selecionado):
        bib = carregar_biblioteca()
        if nome_selecionado in bib:
            dados = bib[nome_selecionado]
            return dados.get("audio_path"), dados.get("transcript")
        return None, ""

    dropdown_voz.change(fn=sync_dados_voz, inputs=dropdown_voz, outputs=[preview_audio_ref, preview_texto_ref])

    current_job_id = gr.State(value=None)

    def atualizar_tabela_jobs():
        linhas = []
        for job in fila_jobs.list_jobs():
            linhas.append([
                job["id"],
                job["status"],
                job["created_at"],
                job["started_at"] or "",
                job["finished_at"] or "",
                job["error"] or ""
            ])
        return linhas

    def sintetizar_audio_controller(nome_voz, texto, usar_chunking, temperatura):
        if not nome_voz:
            return "<div class='toast-banner toast-error'>Selecione uma voz da biblioteca.</div>", None, atualizar_tabela_jobs()
        if not texto or not texto.strip():
            return "<div class='toast-banner toast-error'>O texto não pode estar vazio.</div>", None, atualizar_tabela_jobs()

        bib = carregar_biblioteca()
        dados_voz = bib.get(nome_voz)
        if not dados_voz:
            return "<div class='toast-banner toast-error'>A voz selecionada não existe mais na biblioteca.</div>", None, atualizar_tabela_jobs()
        if not os.path.isfile(dados_voz.get("audio_path", "")):
            return "<div class='toast-banner toast-error'>O áudio de referência dessa voz não foi encontrado.</div>", None, atualizar_tabela_jobs()

        try:
            job_id = fila_jobs.submit({
                "nome_voz": nome_voz,
                "audio_path": dados_voz["audio_path"],
                "transcricao": dados_voz["transcript"],
                "texto": texto,
                "usar_chunking": bool(usar_chunking),
                "temperatura": float(temperatura)
            })
            registrar_log(f"[JOB {job_id}] Solicitação adicionada à fila.")
            return (
                f"<div class='toast-banner toast-success'>⏳ Solicitação recebida. Job {job_id} aguardando processamento.</div>",
                job_id,
                atualizar_tabela_jobs()
            )
        except Exception:
            registrar_log(f"[ERRO FILA]: {traceback.format_exc()}")
            return "<div class='toast-banner toast-error'>Não foi possível adicionar à fila. Consulte o log.</div>", None, atualizar_tabela_jobs()

    btn_sintetizar.click(
        fn=sintetizar_audio_controller,
        inputs=[dropdown_voz, input_texto_narrador, chk_chunking, slider_temp],
        outputs=[toast_status, current_job_id, tabela_jobs]
    )

    def consultar_job(job_id):
        if not job_id:
            return gr.update(), gr.update(), atualizar_tabela_jobs(), gr.update()

        job = fila_jobs.get(job_id)
        if not job:
            return (
                "<div class='toast-banner toast-error'>O Job não está mais disponível no histórico da fila.</div>",
                gr.update(),
                atualizar_tabela_jobs(),
                atualizar_tabela_historico()
            )

        if job["status"] == "queued":
            mensagem = f"<div class='toast-banner toast-success'>⏳ Job {job_id} aguardando na fila.</div>"
            audio = gr.update()
            historico = gr.update()
        elif job["status"] == "running":
            mensagem = f"<div class='toast-banner toast-success'>⚙️ Job {job_id} em processamento. Aguarde a conclusão.</div>"
            audio = gr.update()
            historico = gr.update()
        elif job["status"] == "completed":
            mensagem = f"<div class='toast-banner toast-success'>✅ Job {job_id} concluído com sucesso!</div>"
            audio = job["result"]
            historico = atualizar_tabela_historico()
        else:
            mensagem = f"<div class='toast-banner toast-error'>❌ Job {job_id} falhou: {job['error'] or 'Erro não informado.'}</div>"
            audio = gr.update()
            historico = gr.update()

        return mensagem, audio, atualizar_tabela_jobs(), historico

    timer_jobs = gr.Timer(2.0, active=True)
    timer_jobs.tick(
        fn=consultar_job,
        inputs=[current_job_id],
        outputs=[toast_status, audio_output, tabela_jobs, tabela_historico]
    )

    def salvar_nova_voz_action(nome, audio, transcricao):
        if not nome or not audio or not transcricao:
            return "<div class='toast-banner toast-error'>Preencha todos os campos.</div>", gr.update(), renderizar_cards_vozes()
        
        bib = carregar_biblioteca()
        ext = Path(audio).suffix
        nome_slug = nome.strip().lower().replace(" ", "_")
        destino = VOICES_DIR / f"voz_{nome_slug}{ext}"
        
        shutil.copy(audio, destino)
        bib[nome] = {"audio_path": str(destino), "transcript": transcricao}
        salvar_json(VOICES_FILE, bib)
        
        return f"<div class='toast-banner toast-success'>Voz '{nome}' salva com sucesso!</div>", gr.update(choices=list(bib.keys()), value=nome), renderizar_cards_vozes()

    btn_salvar_voz.click(
        fn=salvar_nova_voz_action,
        inputs=[novo_nome_voz, novo_audio_ref, nova_transcricao],
        outputs=[status_salvamento, dropdown_voz, cards_html]
    )

    filtro_busca_hist.change(fn=atualizar_tabela_historico, inputs=[filtro_busca_hist, filtro_voz_hist], outputs=tabela_historico)
    filtro_voz_hist.change(fn=atualizar_tabela_historico, inputs=[filtro_busca_hist, filtro_voz_hist], outputs=tabela_historico)

    def carregar_linha_hist(evt: gr.SelectData):
        linha = tabela_historico.value[evt.index[0]]
        return linha[3], linha[3]

    tabela_historico.select(fn=carregar_linha_hist, outputs=[caminho_selecionado, player_historico])

if __name__ == "__main__":
    registrar_log("Servidor pronto com suporte a Chunking e Concatenação.")
    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860, share=False)