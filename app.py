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

# 1. Blindagem de importações locais
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
    return carregar_json(HISTORY_FILE, {})

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

def atualizar_tabela_jobs():
    """Busca os jobs mais recentes na base de dados para exibir na aba de Gerenciamento."""
    jobs_list, _ = job_repo.list(offset=0, limit=50)
    linhas = []
    for j in jobs_list:
        job_id_short = str(j.id)[:8]
        status = j.status
        criado = j.created_at.strftime("%H:%M:%S") if j.created_at else ""
        
        # Extrai detalhes do input ou progresso
        inp = j.input or {}
        texto_resumo = inp.get("text", "")
        if not texto_resumo and "chunks" in inp:
            texto_resumo = f"[Modo Chunks - {len(inp['chunks'])} blocos]"
        if len(texto_resumo) > 35:
            texto_resumo = texto_resumo[:35] + "..."

        # Progresso ou detalhes de saída
        progresso_txt = "-"
        audio_path = ""
        if status == "running" and j.output and isinstance(j.output, dict):
            status_msg = j.output.get("status_msg", "A processar...")
            progresso_txt = f"⚙️ {status_msg}"
        elif status == "succeeded" and j.output and isinstance(j.output, dict):
            audio_path = j.output.get("audio_path", "")
            progresso_txt = "✅ Concluído"
        elif status == "failed":
            progresso_txt = f"❌ {j.error_message or 'Erro'}"
        elif status == "cancelled":
            progresso_txt = "🛑 Cancelado"

        linhas.append([str(j.id), job_id_short, criado, status, progresso_txt, texto_resumo, audio_path])
    return linhas

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
.toast-success { background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.3); color: #22C55E; }
.toast-error { background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); color: #EF4444; }
.toast-info { background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.3); color: #3B82F6; }
"""

# --- Interface Gráfica Gradio ---

with gr.Blocks(title="Qwen3-TTS · Síntese de Voz") as demo:
    
    active_job_id = gr.State(value=None)
    status_timer = gr.Timer(value=2, active=False)

    gr.HTML("""
    <div class='app-header'>
        <div class='header-left'>
            <div class='header-icon'>🎙️</div>
            <div class='header-titles'>
                <h1>Qwen3-TTS</h1>
                <p>Síntese Didática Desacoplada · Gestão por Fila Durável</p>
            </div>
        </div>
        <div class='status-badge'>
            <div class='status-dot'></div>
            <span>Worker Ativo</span>
        </div>
    </div>
    """)

    biblioteca_inicial = carregar_biblioteca()
    nomes_vozes = list(biblioteca_inicial.keys())

    with gr.Tabs():
        
        # --- ABA 1: SÍNTESE (LIMPA, SEM BOTÃO DE CANCELAR) ---
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
                            placeholder="Use colchetes para chunks em inglês: Por isso, a gente diz... [I am watching a series.]",
                            lines=5,
                            value="Por isso, a gente diz... [I am watching a series.] Ou então... [Let's watch the game!]"
                        )
                        with gr.Row():
                            chk_chunking = gr.Checkbox(
                                label="Modo Alta Fidelidade (Chunking por Idioma)",
                                value=True,
                            )
                            slider_temp = gr.Slider(
                                minimum=0.5, maximum=0.8, value=0.65, step=0.05, label="Temperatura"
                            )

                        btn_sintetizar = gr.Button("⚡ Enfileirar Síntese", elem_classes="btn-primary")

                with gr.Column(scale=4):
                    toast_status = gr.HTML("<div class='toast-banner toast-info'>Pronto para enfileirar tarefas. Acompanhe na aba Jobs.</div>")
                    
                    with gr.Group(elem_classes="card-panel"):
                        gr.Markdown("### Último Resultado Gerado")
                        audio_output = gr.Audio(label="", interactive=False)
                    
                    with gr.Accordion("Referência da Voz Ativa", open=False):
                        preview_audio_ref = gr.Audio(label="Áudio Original", interactive=False)
                        preview_texto_ref = gr.Textbox(label="Transcrição", interactive=False, lines=2)

        # --- ABA 2: GERENCIAMENTO DE JOBS (NOVA) ---
        with gr.TabItem("⚙️ Gestão de Jobs"):
            with gr.Group(elem_classes="card-panel"):
                gr.Markdown("### Fila de Tarefas e Histórico de Execução")
                gr.Markdown("Acompanhe o estado de cada tarefa em tempo real. Se o servidor for reiniciado, os jobs pendentes são retomados automaticamente.")
                
                with gr.Row():
                    btn_atualizar_jobs = gr.Button("🔄 Atualizar Lista", scale=1)
                
                tabela_jobs = gr.Dataframe(
                    headers=["UUID Completo", "ID", "Hora", "Status", "Progresso / Detalhes", "Texto", "Caminho do Áudio"],
                    value=atualizar_tabela_jobs(),
                    interactive=False,
                    wrap=True,
                    visible=True
                )
                
                with gr.Row():
                    job_uuid_selecionado = gr.Textbox(label="ID do Job Selecionado", interactive=False, scale=3)
                    btn_cancelar_job = gr.Button("🛑 Cancelar Job", variant="stop", scale=1)
                    btn_deletar_job = gr.Button("🗑️ Deletar Job & Áudio", variant="stop", scale=1)

                with gr.Row():
                    player_job = gr.Audio(label="Ouvir Áudio do Job Selecionado", interactive=False)

        # --- ABA 3: CADASTRAR NOVA VOZ ---
        with gr.TabItem("➕ Cadastrar Nova Voz"):
            with gr.Group(elem_classes="card-panel"):
                gr.Markdown("### Cadastro de Nova Matriz de Voz")
                with gr.Row():
                    with gr.Column():
                        novo_audio_ref = gr.Audio(type="filepath", label="Áudio de Referência (WAV)", sources=["upload", "microphone"])
                    with gr.Column():
                        nova_transcricao = gr.Textbox(label="Transcrição Exata do Áudio", lines=2)
                        novo_nome_voz = gr.Textbox(label="Nome do Narrador")
                        btn_salvar_voz = gr.Button("Salvar Voz na Biblioteca", elem_classes="btn-primary")
                        status_salvamento = gr.HTML("")

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

    # Enfileirar Síntese
    def enfileirar_pedido_sintese(nome_voz: str, texto: str, usar_chunking: bool, temperatura: float):
        if not nome_voz or not texto or not texto.strip():
            return "<div class='toast-banner toast-error'>❌ Preencha a voz e o texto corretamente.</div>", None, gr.Timer(active=False), atualizar_tabela_jobs()

        bib = carregar_biblioteca()
        if nome_voz not in bib:
            return "<div class='toast-banner toast-error'>❌ Voz não encontrada.</div>", None, gr.Timer(active=False), atualizar_tabela_jobs()

        dados_voz = bib[nome_voz]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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

        job = job_repo.enqueue(kind="tts_generate", input=job_input)
        toast = f"<div class='toast-banner toast-info'>⏳ Job {str(job.id)[:8]} enfileirado com sucesso! Vá para a aba 'Gestão de Jobs' para acompanhar.</div>"
        return toast, str(job.id), gr.Timer(active=True), atualizar_tabela_jobs()

    btn_sintetizar.click(
        fn=enfileirar_pedido_sintese,
        inputs=[dropdown_voz, input_texto_narrador, chk_chunking, slider_temp],
        outputs=[toast_status, active_job_id, status_timer, tabela_jobs]
    )

<<<<<<< HEAD
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
=======
    # Polling global do job ativo para atualizar o player na aba de síntese quando concluído
    def checar_status_ativo(job_id_str: str | None):
>>>>>>> b83f554 (ok)
        if not job_id_str:
            return gr.update(), gr.Timer(active=False), atualizar_tabela_jobs()
        
        job = job_repo.get(job_id_str)
        if not job:
            return gr.update(), gr.Timer(active=False), atualizar_tabela_jobs()

<<<<<<< HEAD
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
=======
        tabela = atualizar_tabela_jobs()
        if job.status == "succeeded":
            caminho = job.output.get("audio_path") if job.output else None
            return caminho, gr.Timer(active=False), tabela
        elif job.status in ["failed", "cancelled"]:
            return gr.update(), gr.Timer(active=False), tabela

        return gr.update(), gr.Timer(active=True), tabela
>>>>>>> b83f554 (ok)

    status_timer.tick(
        fn=checar_status_ativo,
        inputs=[active_job_id],
        outputs=[audio_output, status_timer, tabela_jobs]
    )

    # Selecionar linha na tabela de Jobs para ouvir ou gerenciar
    def selecionar_job_tabela(evt: gr.SelectData):
        tabela_dados = tabela_jobs.value
        if not tabela_dados or evt.index[0] >= len(tabela_dados):
            return "", None
        linha = tabela_dados[evt.index[0]]
        full_uuid = linha[0]
        audio_path = linha[6]
        
        caminho_audio = audio_path if (audio_path and os.path.exists(audio_path)) else None
        return full_uuid, caminho_audio

    tabela_jobs.select(
        fn=selecionar_job_tabela,
        outputs=[job_uuid_selecionado, player_job]
    )

    # Ações na Gestão de Jobs
    def acao_cancelar_job(job_id_str: str):
        if not job_id_str:
            return "Selecione um job na tabela.", atualizar_tabela_jobs()
        job_repo.request_cancellation(job_id=job_id_str, now=datetime.now(timezone.utc))
        return f"Cancelamento solicitado para o job {job_id_str[:8]}", atualizar_tabela_jobs()

    btn_cancelar_job.click(
        fn=acao_cancelar_job,
        inputs=[job_uuid_selecionado],
        outputs=[toast_status, tabela_jobs]
    )

    # Exclusão em Cascata (Deleta o Job e apaga o arquivo de áudio associado)
    def acao_deletar_job(job_id_str: str):
        if not job_id_str:
            return "Selecione um job na tabela.", atualizar_tabela_jobs(), None
        
        job = job_repo.get(job_id_str)
        if job and job.output and isinstance(job.output, dict):
            audio_p = job.output.get("audio_path")
            if audio_p and os.path.exists(audio_p):
                try:
                    os.remove(audio_p)
                except Exception:
                    pass

        # Remove fisicamente do SQLite (executando comando SQL direto no repositório)
        try:
            with job_repo._get_conn() as conn:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id_str,))
        except Exception:
            pass

        return f"Job {job_id_str[:8]} e respetivo áudio eliminados.", atualizar_tabela_jobs(), None

    btn_deletar_job.click(
        fn=acao_deletar_job,
        inputs=[job_uuid_selecionado],
        outputs=[toast_status, tabela_jobs, player_job]
    )

    btn_atualizar_jobs.click(
        fn=atualizar_tabela_jobs,
        outputs=[tabela_jobs]
    )

    # Cadastrar nova voz
    def salvar_nova_voz_action(nome: str, audio: str, transcricao: str):
        if not nome or not audio or not transcricao:
            return "<div class='toast-banner toast-error'>Preencha todos os campos.</div>", gr.update()
        bib = carregar_biblioteca()
        ext = Path(audio).suffix
        nome_slug = nome.strip().lower().replace(" ", "_")
        destino = VOICES_DIR / f"voz_{nome_slug}{ext}"
        shutil.copy(audio, destino)
        bib[nome] = {"audio_path": str(destino), "transcript": transcricao}
        salvar_json(VOICES_FILE, bib)
        return f"<div class='toast-banner toast-success'>Voz '{nome}' cadastrada!</div>", gr.update(choices=list(bib.keys()), value=nome)

    btn_salvar_voz.click(
        fn=salvar_nova_voz_action,
        inputs=[novo_nome_voz, novo_audio_ref, nova_transcricao],
        outputs=[status_salvamento, dropdown_voz]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", share=False, css=CUSTOM_CSS)