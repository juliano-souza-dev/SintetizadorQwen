from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone

UTC = timezone.utc

# 1. Blindagem de importações locais: garante que a raiz do projeto esteja no topo do sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import gradio as gr
from sqlite_job_repository import SqliteJobRepository

# 2. Configurações de diretórios e arquivos de dados
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
VOICES_DIR = DATA_DIR / "voices"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
VOICES_DIR.mkdir(parents=True, exist_ok=True)

VOICES_FILE = DATA_DIR / "voices.json"
HISTORY_FILE = DATA_DIR / "history.json"

# Instanciação do repositório SQLite compartilhado com o worker
job_repo = SqliteJobRepository(db_path=str(DATA_DIR / "jobs.db"))

# --- Funções Auxiliares de Dados ---

def carregar_json(arquivo: Path | str, padrao: dict | list) -> dict | list:
    caminho = Path(arquivo)
    if caminho.exists():
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return padrao

def salvar_json(arquivo: Path | str, dados: dict | list) -> None:
    try:
        with open(arquivo, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=4)
    except Exception:
        pass

def carregar_biblioteca() -> dict:
    return carregar_json(VOICES_FILE, {})

def carregar_historico() -> list:
    return carregar_json(HISTORY_FILE, [])

def registrar_no_historico(voz_nome: str, texto_falado: str, caminho_arquivo: str | None) -> list:
    historico = carregar_historico()
    novo_registro = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "voz": voz_nome,
        "texto": texto_falado,
        "arquivo": str(caminho_arquivo or "")
    }
    historico.insert(0, novo_registro)
    salvar_json(HISTORY_FILE, historico)
    return atualizar_tabela_historico()

def atualizar_tabela_historico(busca_texto: str = "", filtro_voz: str = "Todas") -> list:
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

def parse_texto_para_chunks(texto: str) -> list[dict]:
    """Divide o texto em blocos de português e inglês com base nos colchetes [exemplo]."""
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

# --- Estilização Visual Customizada ---

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background-color: #0F1115 !important;
    color: #E5E7EB !important;
}

.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background-color: #171A21;
    border: 1px solid #262B36;
    border-radius: 12px;
    margin-bottom: 20px;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-icon {
    width: 44px;
    height: 44px;
    background: rgba(249, 115, 22, 0.15);
    border: 1px solid rgba(249, 115, 22, 0.3);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}
.header-titles h1 {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0;
}
.header-titles p {
    font-size: 13px;
    color: #9CA3AF;
    margin: 0;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 20px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22C55E;
    font-size: 13px;
    font-weight: 500;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #22C55E;
    box-shadow: 0 0 8px #22C55E;
}

.tab-nav {
    background-color: #171A21 !important;
    border: 1px solid #262B36 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 6px !important;
    display: flex !important;
}
.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #9CA3AF !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    min-height: 44px !important;
}
.tab-nav button.selected {
    background-color: #F97316 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

.card-panel {
    background-color: #171A21 !important;
    border: 1px solid #262B36 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

textarea, input[type="text"] {
    background-color: #0F1115 !important;
    border: 1px solid #262B36 !important;
    color: #E5E7EB !important;
    border-radius: 8px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #F97316 !important;
}

.btn-primary {
    background-color: #F97316 !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    min-height: 44px !important;
}
.btn-primary:hover {
    background-color: #EA580C !important;
}

.toast-banner {
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 12px;
}
.toast-success {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #22C55E;
}
.toast-error {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #EF4444;
}
.toast-info {
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #3B82F6;
}

.app-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid #262B36;
    font-size: 12px;
    color: #6B7280;
}
"""

# --- Interface Gráfica Gradio ---

with gr.Blocks(title="Qwen3-TTS · Síntese de Voz") as demo:    
    # Estados da sessão
    active_job_id = gr.State(value=None)
    active_job_meta = gr.State(value={})

    # Timer para monitorar o status do Job a cada 2 segundos
    status_timer = gr.Timer(value=2, active=False)

    # 1. Cabeçalho
    gr.HTML("""
    <div class='app-header'>
        <div class='header-left'>
            <div class='header-icon'>🎙️</div>
            <div class='header-titles'>
                <h1>Qwen3-TTS</h1>
                <p>Síntese Didática Desacoplada · Fila Durável de Tarefas</p>
            </div>
        </div>
        <div class='status-badge'>
            <div class='status-dot'></div>
            <span>Worker Conectado</span>
        </div>
    </div>
    """)

    biblioteca_inicial = carregar_biblioteca()
    nomes_vozes = list(biblioteca_inicial.keys())
    total_hist = len(carregar_historico())

    with gr.Tabs():
        
        # --- ABA 1: SÍNTESE ---
        with gr.TabItem("🎧 Sintetizar Frase"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="card-panel"):
                        gr.Markdown("### 1. Voz do Narrador")
                        dropdown_voz = gr.Dropdown(
                            choices=nomes_vozes,
                            value=nomes_vozes[0] if nomes_vozes else None,
                            label="Selecione a Voz",
                            interactive=True
                        )

                    with gr.Group(elem_classes="card-panel"):
                        gr.Markdown("### 2. Texto para Locução")
                        input_texto_narrador = gr.Textbox(
                            label="",
                            placeholder="Dica: Use colchetes para chunks em inglês: Por isso, a gente diz... [I am watching a series.]",
                            lines=5,
                            value="Por isso, a gente diz... [I am watching a series.] Ou então... [Let's watch the game!] — A imagem se mexe, e você acompanha."
                        )
                        with gr.Row():
                            chk_chunking = gr.Checkbox(
                                label="Modo Alta Fidelidade (Chunking por Idioma)",
                                value=True,
                                info="Divide em blocos PT e EN nativos. Mais natural na troca de língua."
                            )
                            slider_temp = gr.Slider(
                                minimum=0.5,
                                maximum=0.8,
                                value=0.65,
                                step=0.05,
                                label="Temperatura (0.65 ideal)"
                            )

                        with gr.Row():
                            btn_sintetizar = gr.Button("⚡ Enfileirar Síntese", elem_classes="btn-primary", scale=4)
                            btn_cancelar = gr.Button("🛑 Cancelar Pedido", variant="stop", scale=2)

                with gr.Column(scale=4):
                    toast_status = gr.HTML("<div class='toast-banner toast-info'>Pronto para enfileirar tarefas.</div>")
                    
                    with gr.Group(elem_classes="card-panel"):
                        gr.Markdown("### Resultado da Síntese")
                        audio_output = gr.Audio(label="", interactive=False)
                    
                    with gr.Accordion("Referência da Voz Ativa", open=False):
                        preview_audio_ref = gr.Audio(label="Áudio Original", interactive=False)
                        preview_texto_ref = gr.Textbox(label="Transcrição", interactive=False, lines=2)

        # --- ABA 2: CADASTRAR NOVA VOZ ---
        with gr.TabItem("➕ Cadastrar Nova Voz"):
            with gr.Group(elem_classes="card-panel"):
                gr.Markdown("### Cadastro de Nova Matriz de Voz (3 a 5 segundos)")
                with gr.Row():
                    with gr.Column():
                        novo_audio_ref = gr.Audio(type="filepath", label="Áudio de Referência (WAV)", sources=["upload", "microphone"])
                    with gr.Column():
                        nova_transcricao = gr.Textbox(label="Transcrição Exata do Áudio", lines=2)
                        novo_nome_voz = gr.Textbox(label="Nome do Narrador")
                        btn_salvar_voz = gr.Button("Salvar Voz na Biblioteca", elem_classes="btn-primary")
                        status_salvamento = gr.HTML("")

        # --- ABA 3: HISTÓRICO ---
        with gr.TabItem(f"📜 Histórico ({total_hist})"):
            with gr.Group(elem_classes="card-panel"):
                with gr.Row():
                    filtro_busca_hist = gr.Textbox(label="Buscar texto", placeholder="Filtrar...", scale=4)
                    filtro_voz_hist = gr.Dropdown(choices=["Todas"] + nomes_vozes, value="Todas", label="Voz", scale=2)

                tabela_historico = gr.Dataframe(
                    headers=["Data/Hora", "Voz", "Texto", "Arquivo Salvo"],
                    value=atualizar_tabela_historico(),
                    interactive=False,
                    wrap=True
                )
                with gr.Row():
                    caminho_selecionado = gr.Textbox(label="Arquivo Selecionado", interactive=False)
                    player_historico = gr.Audio(label="Ouvir Gravação", interactive=False)

    gr.HTML("""
    <div class='app-footer'>
        <div>Qwen3-TTS Service · Desacoplamento via Fila Durável SQLite</div>
        <div>Pronto para Aulas e Reels Didáticos</div>
    </div>
    """)

    # --- Lógica de Interação ---

    def sync_dados_voz(nome_selecionado: str):
        bib = carregar_biblioteca()
        if nome_selecionado in bib:
            dados = bib[nome_selecionado]
            return dados.get("audio_path"), dados.get("transcript")
        return None, ""

    dropdown_voz.change(
        fn=sync_dados_voz,
        inputs=dropdown_voz,
        outputs=[preview_audio_ref, preview_texto_ref]
    )

    # 1. Enfileirar Pedido de Síntese
    def enfileirar_pedido_sintese(nome_voz: str, texto: str, usar_chunking: bool, temperatura: float):
        if not nome_voz:
            return (
                "<div class='toast-banner toast-error'>❌ Selecione uma voz da biblioteca.</div>",
                None,
                {},
                gr.Timer(active=False)
            )
        if not texto or not texto.strip():
            return (
                "<div class='toast-banner toast-error'>❌ O texto do narrador não pode estar vazio.</div>",
                None,
                {},
                gr.Timer(active=False)
            )

        bib = carregar_biblioteca()
        if nome_voz not in bib:
            return (
                "<div class='toast-banner toast-error'>❌ Voz selecionada não encontrada na biblioteca.</div>",
                None,
                {},
                gr.Timer(active=False)
            )

        dados_voz = bib[nome_voz]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arquivo_saida = str(OUTPUTS_DIR / f"audio_{timestamp}.wav")

        job_input = {
            "ref_audio_path": dados_voz["audio_path"],
            "ref_text": dados_voz["transcript"],
            "output_path": arquivo_saida,
            "temperature": temperatura
        }

        if usar_chunking and "[" in texto and "]" in texto:
            job_input["chunks"] = parse_texto_para_chunks(texto)
        else:
            job_input["text"] = texto.replace("[", "").replace("]", "")

        # Enfileira no banco durável compartilhado
        job = job_repo.enqueue(kind="tts_generate", input=job_input)
        meta = {"voz": nome_voz, "texto": texto}

        toast = f"<div class='toast-banner toast-info'>⏳ Tarefa enfileirada ({str(job.id)[:8]}). O Worker assumirá a síntese...</div>"
        return toast, str(job.id), meta, gr.Timer(active=True)

    btn_sintetizar.click(
        fn=enfileirar_pedido_sintese,
        inputs=[dropdown_voz, input_texto_narrador, chk_chunking, slider_temp],
        outputs=[toast_status, active_job_id, active_job_meta, status_timer]
    )

    # 2. Polling de Status do Job (Timer a cada 2s)
    def _html_seguro(valor):
        import html
        return html.escape(str(valor or ""))

    def _detalhes_job(job, mensagem, classe="toast-info"):
        entrada = job.input or {}
        texto_job = entrada.get("text") or " ".join(
            chunk.get("text", "") for chunk in entrada.get("chunks", [])
        )
        modo = "Chunking por idioma" if entrada.get("chunks") else "Síntese direta"
        criado = job.created_at.astimezone().strftime("%d/%m/%Y %H:%M:%S") if job.created_at else "Não informado"
        atualizado = job.updated_at.astimezone().strftime("%d/%m/%Y %H:%M:%S") if job.updated_at else "Não informado"
        worker = job.worker_id or ("Aguardando worker" if job.status == "queued" else "Não identificado")
        erro = (
            f"<p style='color:#F87171;margin:8px 0 0'><b>Erro:</b> {_html_seguro(job.error_message)}</p>"
            if job.error_message else ""
        )
        return f"""
        <div class='toast-banner {classe}' style='margin-bottom:10px'>{mensagem}</div>
        <div style='background:#171A21;border:1px solid #262B36;border-radius:10px;padding:14px;color:#E5E7EB'>
          <div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap'>
            <b>Detalhes da execução</b><span style='color:#9CA3AF'>ID: {_html_seguro(str(job.id))}</span>
          </div>
          <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px'>
            <div><small style='color:#9CA3AF'>Status</small><div><b>{_html_seguro(job.status.upper())}</b></div></div>
            <div><small style='color:#9CA3AF'>Tentativa</small><div><b>{job.attempt}/{job.max_attempts}</b></div></div>
            <div><small style='color:#9CA3AF'>Worker</small><div><b>{_html_seguro(worker)}</b></div></div>
            <div><small style='color:#9CA3AF'>Modo</small><div><b>{_html_seguro(modo)}</b></div></div>
            <div><small style='color:#9CA3AF'>Criado em</small><div><b>{_html_seguro(criado)}</b></div></div>
            <div><small style='color:#9CA3AF'>Última atualização</small><div><b>{_html_seguro(atualizado)}</b></div></div>
          </div>
          <div style='margin-top:12px'><small style='color:#9CA3AF'>Texto enviado para síntese</small>
            <div style='white-space:pre-wrap;max-height:110px;overflow:auto;margin-top:4px'>{_html_seguro(texto_job)}</div>
          </div>
          {erro}
        </div>
        """

    def checar_status_job(job_id_str: str | None, meta: dict):
        if not job_id_str:
            return gr.update(), None, atualizar_tabela_historico(), gr.Timer(active=False)

        job = job_repo.get(job_id_str)
        if not job:
            return "<div class='toast-banner toast-error'>Tarefa não encontrada no banco.</div>", None, atualizar_tabela_historico(), gr.Timer(active=False)

        if job.status == "queued":
            return (
                _detalhes_job(job, "⏳ Na fila. O worker ainda não assumiu esta tarefa."),
                None,
                gr.update(),
                gr.Timer(active=True)
            )

        if job.status == "running":
            return (
                _detalhes_job(job, "⚙️ WORKER ATIVO: tarefa em processamento. A última atualização confirma a atividade do worker."),
                None,
                gr.update(),
                gr.Timer(active=True)
            )

        if job.status == "succeeded":
            caminho_final = job.output.get("audio_path") if job.output else None
            tabela = registrar_no_historico(
                meta.get("voz", "Voz"),
                meta.get("texto", ""),
                caminho_final
            )
            return (
                _detalhes_job(job, "✅ Síntese concluída com sucesso!", "toast-success"),
                caminho_final,
                tabela,
                gr.Timer(active=False)
            )

        if job.status == "failed":
            return (
                _detalhes_job(job, "❌ A síntese falhou. Consulte a mensagem de erro abaixo.", "toast-error"),
                None,
                atualizar_tabela_historico(),
                gr.Timer(active=False)
            )

        if job.status == "cancelled":
            return (
                _detalhes_job(job, "🛑 Tarefa cancelada.", "toast-error"),
                None,
                atualizar_tabela_historico(),
                gr.Timer(active=False)
            )

        return _detalhes_job(job, f"Status atual: {job.status}"), None, atualizar_tabela_historico(), gr.Timer(active=True)

    status_timer.tick(
        fn=checar_status_job,
        inputs=[active_job_id, active_job_meta],
        outputs=[toast_status, audio_output, tabela_historico, status_timer]
    )

    # 3. Cancelar Tarefa
    def cancelar_pedido_atual(job_id_str: str | None):
        if not job_id_str:
            return "<div class='toast-banner toast-error'>Nenhuma tarefa ativa para cancelar.</div>"
        job_repo.request_cancellation(job_id=job_id_str, now=datetime.now(timezone.utc))
        return f"<div class='toast-banner toast-info'>Cancelamento solicitado para a tarefa {job_id_str[:8]}...</div>"

    btn_cancelar.click(
        fn=cancelar_pedido_atual,
        inputs=[active_job_id],
        outputs=[toast_status]
    )

    # 4. Cadastrar Nova Voz
    def salvar_nova_voz_action(nome: str, audio: str, transcricao: str):
        if not nome or not audio or not transcricao:
            return "<div class='toast-banner toast-error'>Preencha todos os campos do cadastro.</div>", gr.update()
        
        bib = carregar_biblioteca()
        ext = Path(audio).suffix
        nome_slug = nome.strip().lower().replace(" ", "_")
        destino = VOICES_DIR / f"voz_{nome_slug}{ext}"
        
        shutil.copy(audio, destino)
        bib[nome] = {"audio_path": str(destino), "transcript": transcricao}
        salvar_json(VOICES_FILE, bib)
        
        return f"<div class='toast-banner toast-success'>Voz '{nome}' cadastrada com sucesso!</div>", gr.update(choices=list(bib.keys()), value=nome)

    btn_salvar_voz.click(
        fn=salvar_nova_voz_action,
        inputs=[novo_nome_voz, novo_audio_ref, nova_transcricao],
        outputs=[status_salvamento, dropdown_voz]
    )

    # 5. Filtros e Seleção do Histórico
    filtro_busca_hist.change(
        fn=atualizar_tabela_historico,
        inputs=[filtro_busca_hist, filtro_voz_hist],
        outputs=tabela_historico
    )
    filtro_voz_hist.change(
        fn=atualizar_tabela_historico,
        inputs=[filtro_busca_hist, filtro_voz_hist],
        outputs=tabela_historico
    )

    def carregar_linha_hist(evt: gr.SelectData):
        linha = tabela_historico.value[evt.index[0]]
        return linha[3], linha[3]

    tabela_historico.select(
        fn=carregar_linha_hist,
        outputs=[caminho_selecionado, player_historico]
    )


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        share=False,
        css=CUSTOM_CSS,
    )