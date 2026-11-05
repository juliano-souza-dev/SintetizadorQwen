import os
import sys

# Garante que a pasta atual seja encontrada pelo Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Desativa o Xet para evitar travamentos de leitura no Windows
os.environ["HF_HUB_DISABLE_XET"] = "1"

# IMPORTANTE: Manter a mesma ordem de importacao da versao que funcionou
# Carrega as DLLs do PyTorch e Audio na memoria ANTES do Gradio
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

CAMINHO_MODELO = r"C:\Users\jsouz\Projects\nova-generator\models\qwen3-tts\Qwen3-TTS-12Hz-1.7B-Base"
VOICES_FILE = "voices.json"
HISTORY_FILE = "history.json"
HISTORICO_DIR = Path("historico_audios")
HISTORICO_DIR.mkdir(exist_ok=True)

# Carrega o modelo na Main Thread ANTES de subir o Gradio (evita o erro -1073741819)
tts_engine = TTSService(CAMINHO_MODELO)

def carregar_json(arquivo, padrao):
    if os.path.exists(arquivo):
        try:
            with open(arquivo, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            registrar_log(f"[ERRO JSON] Falha ao ler {arquivo}: {e}")
    return padrao

def salvar_json(arquivo, dados):
    try:
        with open(arquivo, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=4)
    except Exception as e:
        registrar_log(f"[ERRO JSON] Falha ao salvar {arquivo}: {e}")

def carregar_biblioteca():
    return carregar_json(VOICES_FILE, {})

def carregar_historico():
    return carregar_json(HISTORY_FILE, [])

def registrar_no_historico(voz_nome, texto_falado, arquivo_gerado_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ext = Path(arquivo_gerado_path).suffix
    novo_nome = HISTORICO_DIR / f"gerado_{timestamp}{ext}"
    
    shutil.copy(arquivo_gerado_path, novo_nome)
    
    historico = carregar_historico()
    novo_registro = {
        "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "voz": voz_nome,
        "texto": texto_falado,
        "arquivo": str(novo_nome)
    }
    
    historico.insert(0, novo_registro)
    salvar_json(HISTORY_FILE, historico)
    return atualizar_lista_historico()

def atualizar_lista_historico():
    historico = carregar_historico()
    return [[h["data"], h["voz"], h["texto"], h["arquivo"]] for h in historico]

def salvar_na_biblioteca(nome_voz, audio_path, transcript):
    if not nome_voz or not audio_path or not transcript:
        return "Erro: Preencha o nome, envie o audio e informe a transcricao.", gr.update(choices=list(carregar_biblioteca().keys()))
    
    biblioteca = carregar_biblioteca()
    ext = Path(audio_path).suffix
    novo_nome_audio = f"voz_{nome_voz.strip().lower().replace(' ', '_')}{ext}"
    
    shutil.copy(audio_path, novo_nome_audio)
    
    biblioteca[nome_voz] = {
        "audio_path": novo_nome_audio,
        "transcript": transcript
    }
    
    salvar_json(VOICES_FILE, biblioteca)
    registrar_log(f"Nova voz salva na biblioteca: {nome_voz}")
    chaves = list(biblioteca.keys())
    return f"Voz '{nome_voz}' salva com sucesso!", gr.update(choices=chaves, value=nome_voz)

def atualizar_campos_voz(nome_selecionado):
    biblioteca = carregar_biblioteca()
    if nome_selecionado in biblioteca:
        dados = biblioteca[nome_selecionado]
        return dados["audio_path"], dados["transcript"]
    return None, ""

def gerar_audio_biblioteca(nome_voz, texto_falado):
    biblioteca = carregar_biblioteca()
    if nome_voz not in biblioteca:
        return "Erro: Selecione uma voz valida da biblioteca.", None, atualizar_lista_historico()
    
    dados = biblioteca[nome_voz]
    try:
        output = tts_engine.synthesize(
            ref_audio_path=dados["audio_path"],
            ref_text=dados["transcript"],
            text_to_speak=texto_falado
        )
        tabela_atualizada = registrar_no_historico(nome_voz, texto_falado, output)
        return "Audio gerado com sucesso!", output, tabela_atualizada
    except Exception as e:
        registrar_log(f"[ERRO INTERFACE]:\n{traceback.format_exc()}")
        return f"Erro na geracao: {str(e)}", None, atualizar_lista_historico()

biblioteca_inicial = carregar_biblioteca()
nomes_vozes = list(biblioteca_inicial.keys())

with gr.Blocks(title="Sintetizador Qwen3-TTS") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS - Biblioteca, Síntese e Histórico (CPU)")
    
    with gr.Tabs():
        with gr.TabItem("🎧 Sintetizar Nova Frase"):
            with gr.Row():
                with gr.Column():
                    dropdown_voz = gr.Dropdown(choices=nomes_vozes, label="Selecione a Voz na Biblioteca", value=nomes_vozes[0] if nomes_vozes else None)
                    preview_audio = gr.Audio(label="Áudio de Referência Vinculado", interactive=False)
                    preview_transcricao = gr.Textbox(label="Transcrição da Voz", interactive=False, lines=2)
                    texto_novo = gr.Textbox(label="O que o narrador deve falar?", lines=3, value="Guys... presta atenção nisso aqui. No dia a dia a gente fala... I'm gonna do it.")
                    botao_sintetizar = gr.Button("Gerar Áudio", variant="primary")
                with gr.Column():
                    status_sintese = gr.Textbox(label="Status")
                    output_sintese = gr.Audio(label="Resultado")

            dropdown_voz.change(fn=atualizar_campos_voz, inputs=dropdown_voz, outputs=[preview_audio, preview_transcricao])

        with gr.TabItem("➕ Cadastrar Nova Voz"):
            with gr.Row():
                with gr.Column():
                    input_nome_novo = gr.Textbox(label="Nome da Voz")
                    input_audio_novo = gr.Audio(type="filepath", label="Áudio de Referência (WAV ou MP3)")
                    input_texto_novo = gr.Textbox(label="Transcrição Exata do Áudio", lines=2)
                    botao_salvar_voz = gr.Button("Salvar na Biblioteca", variant="secondary")
                with gr.Column():
                    status_cadastro = gr.Textbox(label="Status do Cadastro")

            botao_salvar_voz.click(
                fn=salvar_na_biblioteca,
                inputs=[input_nome_novo, input_audio_novo, input_texto_novo],
                outputs=[status_cadastro, dropdown_voz]
            )

        with gr.TabItem("📜 Histórico de Gerações"):
            tabela_historico = gr.Dataframe(
                headers=["Data/Hora", "Voz Utilizada", "Texto Falado", "Arquivo Salvo"],
                value=atualizar_lista_historico(),
                interactive=False,
                wrap=True
            )
            
            with gr.Row():
                input_caminho_reproduzir = gr.Textbox(label="Caminho do Arquivo para Ouvir")
                audio_historico_player = gr.Audio(label="Player do Histórico")
            
            def selecionar_historico(evt: gr.SelectData):
                row_values = tabela_historico.value[evt.index[0]]
                caminho_arquivo = row_values[3]
                return caminho_arquivo, caminho_arquivo

            tabela_historico.select(fn=selecionar_historico, outputs=[input_caminho_reproduzir, audio_historico_player])

    botao_sintetizar.click(
        fn=gerar_audio_biblioteca,
        inputs=[dropdown_voz, texto_novo],
        outputs=[status_sintese, output_sintese, tabela_historico]
    )

if __name__ == "__main__":
    registrar_log("Iniciando o servidor Gradio em http://127.0.0.1:7860 ...")
    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860, share=False)