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

def _formatar_duracao(segundos):
    if segundos is None or segundos < 0: return "Calculando"
    segundos = int(round(segundos))
    if segundos < 60: return f"{segundos}s"
    minutos, resto = divmod(segundos, 60)
    if minutos < 60: return f"{minutos}min {resto:02d}s"
    horas, minutos = divmod(minutos, 60)
    return f"{horas}h {minutos:02d}min"

def _estimativas_progresso(job):
    progress = job.progress or {}
    history = progress.get("chunk_history") or []
    valid = [item for item in history if isinstance(item, dict) and float(item.get("text_chars") or 0) > 0 and float(item.get("duration_seconds") or 0) > 0]
    total_chars = sum(float(item["text_chars"]) for item in valid)
    total_seconds = sum(float(item["duration_seconds"]) for item in valid)
    rate = total_seconds / total_chars if total_chars else None
    elapsed_current = 0.0
    started = progress.get("chunk_started_at")
    if started:
        try:
            started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            elapsed_current = max(0.0, (datetime.now(timezone.utc) - started_dt).total_seconds())
        except (TypeError, ValueError): pass
    chunk_eta = task_eta = None
    chunk_chars = float(progress.get("chunk_chars") or 0)
    remaining_chars = float(progress.get("remaining_chars") or 0)
    if rate and progress.get("phase") == "synthesizing" and started:
        chunk_eta = max(0.0, rate * chunk_chars - elapsed_current)
        task_eta = max(0.0, rate * remaining_chars - elapsed_current)
    elif rate and progress.get("phase") == "synthesizing":
        task_eta = max(0.0, rate * remaining_chars)
    return chunk_eta, task_eta, elapsed_current, rate

def _resumo_progresso(job):
    progress = job.progress or {}
    if job.status == "queued": return "Aguardando na fila", "—"
    if job.status == "retryable": return "Aguardando nova tentativa", "—"
    if job.status == "succeeded": return "Concluído", "100%"
    if job.status == "failed": return "Falhou", "—"
    if job.status == "cancelled": return "Cancelado", "—"
    phase = progress.get("phase")
    if phase == "synthesizing":
        current, total = int(progress.get("current_chunk") or 0), int(progress.get("total_chunks") or 0)
        completed = int(progress.get("completed_chunks") or 0)
        chunk_eta, task_eta, _, _ = _estimativas_progresso(job)
        event = progress.get("event")
        if not total:
            etapas_diretas = {
                "reference_preparation_started": "Preparando áudio de referência",
                "reference_preparation_completed": "Referência preparada",
                "inference_started": "Gerando áudio com Qwen",
                "inference_completed": "Geração concluída, preparando WAV",
                "audio_write_started": "Gravando arquivo WAV",
                "audio_write_completed": "Arquivo WAV gravado",
                "direct_completed": "Síntese direta concluída",
            }
            etapa = etapas_diretas.get(event, "Sintetizando")
            duracao = progress.get("inference_seconds")
            previsao = f"Inferência: {_formatar_duracao(float(duracao))}" if duracao is not None else "Tempo da inferência será informado ao concluir"
            return etapa, previsao
        etapa = f"Chunk {current}/{total}" if total else "Sintetizando"
        percentual = f"{round((completed / total) * 100)}%" if total else "—"
        previsao = f"Chunk: {_formatar_duracao(chunk_eta)} · Total: {_formatar_duracao(task_eta)}" if task_eta is not None else "Estimando após o primeiro chunk"
        return etapa, f"{percentual} · {previsao}"
    if phase == "finalizing": return "Montando arquivo de áudio", "Quase concluído"
    return "Preparando motor / iniciando inferência", "Calculando"

def atualizar_tabela_jobs():
    jobs_list, _ = job_repo.list(offset=0, limit=50)
    linhas = []
    for job in jobs_list:
        criado = job.created_at.astimezone().strftime("%H:%M:%S") if job.created_at else ""
        etapa, previsao = _resumo_progresso(job)
        linhas.append([str(job.id)[:8], criado, job.status.upper(), etapa, previsao, f"{job.attempt}/{job.max_attempts}"])
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
    jobs_timer = gr.Timer(value=2, active=True)

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

        # --- ABA 2: MONITORAMENTO E CONTROLE DE JOBS ---
        with gr.TabItem("⚙️ Gestão de Jobs"):
            with gr.Group(elem_classes="card-panel"):
                gr.Markdown("### Central de processamento")
                gr.Markdown("Acompanhe a fila, a etapa atual e as estimativas. O painel atualiza automaticamente a cada 2 segundos.")
                with gr.Row():
                    btn_atualizar_jobs = gr.Button("↻ Atualizar agora", scale=1)
                    gr.Markdown("**Atualização automática ativa** · sem precisar abrir o terminal", scale=3)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=6):
                        gr.Markdown("#### Fila de tarefas")
                        tabela_jobs = gr.Dataframe(headers=["ID", "Criado", "Status", "Etapa atual", "Progresso", "Previsão"], value=atualizar_tabela_jobs(), interactive=False, wrap=True, visible=True)
                        gr.Markdown("<small>Selecione uma tarefa no painel ao lado para ver detalhes e controlar a execução.</small>")
                    with gr.Column(scale=5):
                        gr.Markdown("#### Tarefa selecionada")
                        job_selector = gr.Dropdown(label="Selecionar job", choices=[], value=None, interactive=True, allow_custom_value=False)
                        job_details = gr.HTML("<div class='toast-banner toast-info'>Selecione uma tarefa para acompanhar os detalhes.</div>")
                        with gr.Row():
                            btn_cancelar_job = gr.Button("⏸ Solicitar cancelamento", variant="stop", scale=1)
                            btn_deletar_job = gr.Button("🗑 Excluir job", variant="stop", scale=1)
                        player_job = gr.Audio(label="Áudio gerado", interactive=False)
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

    # Polling do job ativo com detalhes da execução
    def _html_seguro(valor):
        import html
        return html.escape(str(valor or ""))

    def _detalhes_job(job, mensagem, classe="toast-info"):
        entrada = job.input or {}
        texto_job = entrada.get("text") or " ".join(chunk.get("text", "") for chunk in entrada.get("chunks", []))
        modo = "Chunking por idioma" if entrada.get("chunks") else "Síntese direta"
        criado = job.created_at.astimezone().strftime("%d/%m/%Y %H:%M:%S") if job.created_at else "Não informado"
        worker = job.worker_id or ("Aguardando worker" if job.status == "queued" else "Não identificado")
        if job.heartbeat_at:
            heartbeat_age = max(0, int((datetime.now(timezone.utc) - job.heartbeat_at).total_seconds()))
            sinal_worker = f"há {_formatar_duracao(heartbeat_age)}"
        else:
            sinal_worker = "ainda não recebido"
        progress = job.progress or {}
        chunk_eta, task_eta, chunk_elapsed, _ = _estimativas_progresso(job)
        etapa, resumo = _resumo_progresso(job)
        total_chunks = int(progress.get("total_chunks") or len(entrada.get("chunks") or []))
        current_chunk = int(progress.get("current_chunk") or 0)
        completed = int(progress.get("completed_chunks") or 0)
        percentual = min(100, round((completed / total_chunks) * 100)) if total_chunks else (100 if job.status == "succeeded" else 0)
        if job.status == "running" and total_chunks and progress.get("phase") == "synthesizing" and chunk_eta is not None and chunk_elapsed + chunk_eta > 0:
            percentual = min(99, round(((completed + chunk_elapsed / (chunk_elapsed + chunk_eta)) / total_chunks) * 100))
        elapsed_total = max(0, (datetime.now(timezone.utc) - job.started_at).total_seconds()) if job.started_at else None
        elapsed_label = _formatar_duracao(elapsed_total) if elapsed_total is not None else "—"
        chunk_label = f"{current_chunk}/{total_chunks}" if total_chunks else "—"
        chunk_eta_label = _formatar_duracao(chunk_eta) if chunk_eta is not None else "Calculando"
        task_eta_label = _formatar_duracao(task_eta) if task_eta is not None else "Calculando"
        texto_chunk = _html_seguro(progress.get("chunk_text", ""))
        metricas_diretas = []
        for chave, rotulo in (
            ("preparation_seconds", "Preparação da referência"),
            ("inference_seconds", "Inferência Qwen"),
            ("audio_write_seconds", "Gravação do WAV"),
            ("total_seconds", "Tempo total da síntese"),
        ):
            valor = progress.get(chave)
            if isinstance(valor, (int, float)):
                metricas_diretas.append(
                    f"<span>{rotulo}: <b style='color:#E5E7EB'>{_formatar_duracao(float(valor))}</b></span>"
                )
        metricas_html = (
            "<div style='margin-top:12px;padding:10px;background:#101319;border-radius:8px'>"
            "<small style='color:#9CA3AF'>Medições da síntese direta</small>"
            "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-top:6px'>"
            + "".join(metricas_diretas) + "</div></div>"
        ) if metricas_diretas else ""
        audio_tokens = progress.get("generated_audio_tokens")
        audio_frames = progress.get("generated_audio_frames")
        token_rate = progress.get("tokens_per_second")
        token_elapsed = progress.get("inference_elapsed_seconds")
        audio_token_html = ""
        if isinstance(audio_tokens, (int, float)):
            token_rate_label = f"{float(token_rate):.1f} tokens/s" if isinstance(token_rate, (int, float)) else "Calculando"
            frames_label = f"{int(audio_frames):,}".replace(",", ".") if isinstance(audio_frames, (int, float)) else "—"
            token_elapsed_label = _formatar_duracao(float(token_elapsed)) if isinstance(token_elapsed, (int, float)) else "—"
            tokens_label = f"{int(audio_tokens):,}".replace(",", ".")
            audio_token_html = (
                "<div style='margin-top:12px;padding:12px;background:#101319;border:1px solid #303746;border-radius:8px'>"
                "<small style='color:#9CA3AF'>Geração interna do Qwen · dados reais</small>"
                "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:8px'>"
                f"<div><small style='color:#9CA3AF'>Tokens de áudio</small><div style='font-size:20px;font-weight:700'>{tokens_label}</div></div>"
                f"<div><small style='color:#9CA3AF'>Frames gerados</small><div style='font-size:20px;font-weight:700'>{frames_label}</div></div>"
                f"<div><small style='color:#9CA3AF'>Velocidade média</small><div style='font-size:20px;font-weight:700'>{token_rate_label}</div></div>"
                f"<div><small style='color:#9CA3AF'>Tempo de inferência</small><div style='font-size:20px;font-weight:700'>{token_elapsed_label}</div></div>"
                "</div><div style='margin-top:8px;color:#9CA3AF;font-size:11px'>O total de tokens não é conhecido antecipadamente; por isso, não exibimos porcentagem fictícia.</div></div>"
            )
        evento_atual = _html_seguro(progress.get("event", ""))
        evento_html = (
            f"<div style='margin-top:8px;color:#9CA3AF;font-size:12px'>Último evento: <b style='color:#E5E7EB'>{evento_atual}</b></div>"
        ) if evento_atual and not total_chunks else ""
        erro = f"<div style='margin-top:12px;color:#FCA5A5'><b>Erro:</b><pre style='white-space:pre-wrap'>{_html_seguro(job.error_message)}</pre></div>" if job.error_message else ""
        cancelamento = "<div style='margin-top:10px;color:#FBBF24'>Cancelamento solicitado. Aguardando o worker encerrar a etapa atual.</div>" if job.status == "running" and job.cancel_requested_at else ""
        exclusao = "<div style='margin-top:10px;color:#FBBF24'>Exclusão pendente. O registro e os arquivos serão removidos quando o worker parar.</div>" if job.deletion_requested else ""
        chunk_html = f"<div style='margin-top:12px;padding:10px;background:#101319;border-radius:8px'><small style='color:#9CA3AF'>Texto do chunk em processamento</small><div style='margin-top:5px'>{texto_chunk}</div></div>" if texto_chunk else ""
        return f"""
        <div class="toast-banner {classe}" style="margin-bottom:10px">{_html_seguro(mensagem)}</div>
        <div style="background:#171A21;border:1px solid #303746;border-radius:12px;padding:16px;color:#E5E7EB">
          <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;align-items:center">
            <b style="font-size:15px">Execução {str(job.id)[:8]}</b>
            <span style="padding:5px 10px;border-radius:20px;background:#252B36;font-size:12px;font-weight:700">{_html_seguro(job.status.upper())}</span>
          </div>
          <div style="height:9px;background:#303746;border-radius:20px;overflow:hidden;margin-top:14px"><div style="height:100%;width:{percentual}%;background:#F97316;border-radius:20px;transition:width .3s"></div></div>
          <div style="display:flex;justify-content:space-between;margin-top:6px;color:#9CA3AF;font-size:12px"><span>{percentual}% concluído</span><span>{_html_seguro(etapa)}</span></div>
          <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:16px">
            <div style="background:#101319;padding:10px;border-radius:8px"><small style="color:#9CA3AF">Chunk atual</small><div style="font-size:18px;font-weight:700">{chunk_label}</div></div>
            <div style="background:#101319;padding:10px;border-radius:8px"><small style="color:#9CA3AF">Tempo decorrido</small><div style="font-size:18px;font-weight:700">{elapsed_label}</div></div>
            <div style="background:#101319;padding:10px;border-radius:8px"><small style="color:#9CA3AF">Estimativa deste chunk</small><div style="font-size:18px;font-weight:700">{chunk_eta_label}</div></div>
            <div style="background:#101319;padding:10px;border-radius:8px"><small style="color:#9CA3AF">Estimativa da tarefa</small><div style="font-size:18px;font-weight:700">{task_eta_label}</div></div>
          </div>
          <div style="margin-top:14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;color:#9CA3AF;font-size:12px">
            <span>Tentativa: <b style="color:#E5E7EB">{job.attempt}/{job.max_attempts}</b></span><span>Worker: <b style="color:#E5E7EB">{_html_seguro(worker)}</b></span><span>Sinal do worker: <b style="color:#E5E7EB">{_html_seguro(sinal_worker)}</b></span><span>Modo: <b style="color:#E5E7EB">{_html_seguro(modo)}</b></span><span>Criado: <b style="color:#E5E7EB">{_html_seguro(criado)}</b></span>
          </div>
          {chunk_html}
          {evento_html}
          {metricas_html}
          {audio_token_html}
          <div style="margin-top:12px"><small style="color:#9CA3AF">Progresso</small><div>{_html_seguro(resumo)}</div></div>
          {cancelamento}{exclusao}
          <div style="margin-top:12px"><small style="color:#9CA3AF">Texto enviado para síntese</small><div style="white-space:pre-wrap;max-height:100px;overflow:auto;margin-top:4px">{_html_seguro(texto_job)}</div></div>
          {erro}
        </div>
        """

    def checar_status_ativo(job_id_str: str | None):
        if not job_id_str:
            return gr.update(), gr.update(), gr.Timer(active=False), atualizar_tabela_jobs()

        job = job_repo.get(job_id_str)
        if not job:
            return gr.update(), gr.update(), gr.Timer(active=False), atualizar_tabela_jobs()

        tabela = atualizar_tabela_jobs()

        if job.status == "queued":
            return (
                _detalhes_job(job, "⏳ Na fila. Aguardando um worker disponível."),
                gr.update(),
                gr.Timer(active=True),
                tabela
            )

        if job.status == "retryable":
            return (
                _detalhes_job(job, "🔁 A tentativa anterior falhou. O job foi liberado para nova tentativa automática.", "toast-info"),
                gr.update(),
                gr.Timer(active=True),
                tabela
            )

        if job.status == "running":
            return (
                _detalhes_job(job, "⚙️ WORKER ATIVO: tarefa em processamento. A atualização do job indica atividade recente."),
                gr.update(),
                gr.Timer(active=True),
                tabela
            )

        if job.status == "succeeded":
            caminho = job.output.get("audio_path") if job.output and isinstance(job.output, dict) else None
            return (
                _detalhes_job(job, "✅ Síntese concluída com sucesso!", "toast-success"),
                caminho,
                gr.Timer(active=False),
                tabela
            )

        if job.status == "failed":
            return (
                _detalhes_job(job, "❌ A síntese falhou. Consulte a mensagem de erro abaixo.", "toast-error"),
                None,
                gr.Timer(active=False),
                tabela
            )

        if job.status == "cancelled":
            return (
                _detalhes_job(job, "🛑 Tarefa cancelada.", "toast-error"),
                None,
                gr.Timer(active=False),
                tabela
            )

        return (
            _detalhes_job(job, f"Status atual: {_html_seguro(job.status)}"),
            gr.update(),
            gr.Timer(active=True),
            tabela
        )


    status_timer.tick(
        fn=checar_status_ativo,
        inputs=[active_job_id],
        outputs=[toast_status, audio_output, status_timer, tabela_jobs]
    )

    # --- Monitoramento contínuo, seleção e ações seguras dos jobs ---
    def _job_options():
        jobs, _ = job_repo.list(offset=0, limit=50)
        return [(f"{str(job.id)[:8]} · {job.status.upper()} · {(job.created_at.astimezone().strftime('%H:%M:%S') if job.created_at else '--:--:--')}", str(job.id)) for job in jobs]

    def _remover_audio_job(job):
        candidatos = []
        if job.output and isinstance(job.output, dict): candidatos.append(job.output.get("audio_path"))
        if job.input and isinstance(job.input, dict): candidatos.append(job.input.get("output_path"))
        raiz = OUTPUTS_DIR.resolve()
        for candidato in candidatos:
            if not isinstance(candidato, str) or not candidato.strip(): continue
            try:
                caminho = Path(candidato).resolve()
                if os.path.commonpath([str(raiz), str(caminho)]) == str(raiz) and caminho.is_file(): caminho.unlink()
            except (OSError, ValueError): continue

    def processar_exclusoes_pendentes():
        jobs, _ = job_repo.list(offset=0, limit=500)
        for job in jobs:
            if job.deletion_requested and job.status != "running":
                _remover_audio_job(job)
                job_repo.delete_job(job_id=str(job.id))

    def renderizar_job_selecionado(job_id):
        if not job_id: return "<div class='toast-banner toast-info'>Selecione uma tarefa para acompanhar os detalhes.</div>", None, gr.update(interactive=False), gr.update(interactive=False)
        job = job_repo.get(job_id)
        if not job: return "<div class='toast-banner toast-error'>Esta tarefa não existe mais ou já foi excluída.</div>", None, gr.update(interactive=False), gr.update(interactive=False)
        audio_path = None
        if job.status == "succeeded" and job.output and isinstance(job.output, dict):
            candidate = job.output.get("audio_path")
            if candidate and os.path.isfile(candidate): audio_path = candidate
        return _detalhes_job(job, "Acompanhamento atualizado."), audio_path, gr.update(interactive=job.status in ("queued", "retryable", "running")), gr.update(interactive=True)

    def atualizar_monitor_jobs(job_id_atual):
        processar_exclusoes_pendentes()
        tabela = atualizar_tabela_jobs()
        options = _job_options()
        ids = [value for _, value in options]
        selecionado = job_id_atual if job_id_atual in ids else (ids[0] if ids else None)
        detalhes, _, _, _ = renderizar_job_selecionado(selecionado)
        return tabela, gr.update(choices=options, value=selecionado), detalhes

    def acao_cancelar_job(job_id_str):
        if not job_id_str: return "<div class='toast-banner toast-error'>Selecione uma tarefa antes de cancelar.</div>", "<div class='toast-banner toast-info'>Nenhuma tarefa selecionada.</div>", atualizar_tabela_jobs()
        job = job_repo.request_cancellation(job_id=job_id_str, now=datetime.now(timezone.utc))
        if not job: mensagem = "Tarefa não encontrada."
        elif job.status == "running": mensagem = "Cancelamento solicitado. Se o modelo estiver no meio de uma geração, o worker concluirá essa etapa antes de interromper a tarefa."
        elif job.status == "cancelled": mensagem = "Tarefa cancelada."
        else: mensagem = f"Solicitação registrada. Status atual: {job.status}."
        detalhes = _detalhes_job(job, mensagem, "toast-info") if job else mensagem
        return f"<div class='toast-banner toast-info'>{_html_seguro(mensagem)}</div>", detalhes, atualizar_tabela_jobs()

    def acao_deletar_job(job_id_str):
        if not job_id_str: return "<div class='toast-banner toast-error'>Selecione uma tarefa antes de excluir.</div>", atualizar_tabela_jobs(), "<div class='toast-banner toast-info'>Nenhuma tarefa selecionada.</div>", None, None
        job = job_repo.request_deletion(job_id=job_id_str, now=datetime.now(timezone.utc))
        if not job: return "<div class='toast-banner toast-error'>Tarefa não encontrada.</div>", atualizar_tabela_jobs(), "<div class='toast-banner toast-error'>Tarefa não encontrada.</div>", None, None
        estava_rodando = job.status == "running"
        if estava_rodando:
            mensagem = (
                "Cancelamento e exclusão solicitados. A inferência será interrompida "
                "e o job será removido automaticamente quando o worker confirmar "
                "o cancelamento."
            )
            return (
                f"<div class='toast-banner toast-info'>{_html_seguro(mensagem)}</div>",
                atualizar_tabela_jobs(),
                _detalhes_job(job, mensagem, "toast-info"),
                job_id_str,
                None,
            )

        _remover_audio_job(job)
        excluido = job_repo.delete_job(job_id=job_id_str)
        if not excluido:
            mensagem = "Não foi possível excluir o job. Atualize o painel e tente novamente."
            return f"<div class='toast-banner toast-error'>{_html_seguro(mensagem)}</div>", atualizar_tabela_jobs(), _detalhes_job(job, mensagem, "toast-error"), job_id_str, None
        mensagem = "Job e arquivo de áudio excluídos com sucesso."
        return f"<div class='toast-banner toast-success'>{_html_seguro(mensagem)}</div>", atualizar_tabela_jobs(), "<div class='toast-banner toast-info'>Selecione outra tarefa para acompanhar.</div>", None, None

    job_selector.change(fn=renderizar_job_selecionado, inputs=[job_selector], outputs=[job_details, player_job, btn_cancelar_job, btn_deletar_job])
    btn_cancelar_job.click(fn=acao_cancelar_job, inputs=[job_selector], outputs=[toast_status, job_details, tabela_jobs])
    btn_deletar_job.click(fn=acao_deletar_job, inputs=[job_selector], outputs=[toast_status, tabela_jobs, job_details, job_selector, player_job])
    btn_atualizar_jobs.click(fn=atualizar_monitor_jobs, inputs=[job_selector], outputs=[tabela_jobs, job_selector, job_details])
    jobs_timer.tick(fn=atualizar_monitor_jobs, inputs=[job_selector], outputs=[tabela_jobs, job_selector, job_details])
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
    # No modo DEV, o supervisor abre o navegador uma única vez. Assim, reinícios
    # do servidor não abrem novas abas nem solicitam reload explícito da página.
    modo_dev = os.environ.get("QWEN_DEV_MODE") == "1"
    demo.launch(inbrowser=not modo_dev, server_name="127.0.0.1", share=False, css=CUSTOM_CSS)