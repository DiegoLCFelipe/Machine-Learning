import sys, os
import yaml
import pandas as pd

os.system('cls')
sys.path.insert(0,os.path.abspath(os.curdir))
from machine_learning.models.analise_fatorial import matriz_de_correlacoes
from machine_learning.models.analise_fatorial import teste_de_esfericidade_de_bartlett
from machine_learning.models.analise_fatorial import AnaliseFatorial
from machine_learning.utils.logger import LogHandler
import machine_learning.utils.graficos as visualizacao

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
with open(CURRENT_PATH + '/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

PATH = config['path']
VAR_METRICAS = config['variaveis_metricas']
SIGNIFICANCIA = config['significancia']
dados = pd.read_excel(PATH, decimal=',')
dados_analise = dados[VAR_METRICAS]

log = LogHandler()

matriz_corr_pearson = matriz_de_correlacoes(dados_analise)
qui_quadrado, p_valor, graus_de_liberdade = teste_de_esfericidade_de_bartlett(dados_analise, SIGNIFICANCIA)
log.log_info_timestamp(f'Teste de Esfericidade de Bartlet\n\n'
                       f'Chi-quadrado: {qui_quadrado}\n'
                       f'P-valor: {p_valor}\n'
                       f'Graus de liberdade: {graus_de_liberdade}\n')

if p_valor < SIGNIFICANCIA:
      log.log_info(f'Rejeita-se H0 (Mmatriz de correlações igual a matriz identidade):,\n'
            f'logo a análise fatorial pode ser aplicada\n')
      analise_fatorial = AnaliseFatorial(dados_analise, len(VAR_METRICAS))
      fatores_kaiser = analise_fatorial.criterio_de_kaiser()
      analise_fatorial = AnaliseFatorial(dados_analise, fatores_kaiser)
      log.log_info(f'Nº de fatores recomendados utilizando o critério de Kaiser: {fatores_kaiser}\n')
      
      autovalores = analise_fatorial.tabela_de_autovalores()
      cargas = analise_fatorial.tabela_de_cargas()
      comunalidades = analise_fatorial.tabela_de_comunalidades()
      dados_rankeados = analise_fatorial.dados_rankeados()

      autovalores.to_csv(CURRENT_PATH + '/log/autovalores.log', index_label='Fator')
      cargas.to_csv(CURRENT_PATH + '/log/cargas.log', index_label='Variável')
      comunalidades.to_csv(CURRENT_PATH + '/log/comunalidades.log', index_label='Variável')
      dados_rankeados.to_csv(CURRENT_PATH + '/log/dados_rankeados.log')

      visualizacao.mapa_de_calor(matriz_corr_pearson)
      visualizacao.grafico_de_cargas(cargas, 'Fator 1', 'Fator 2')
      visualizacao.fatores_extraidos(autovalores, 'Variância')
      visualizacao.mostrar()
else:
    log.log_warning(f'Aceita-se H0 (Mmatriz de correlações igual a matriz identidade):,\n'
          f'logo a análise fatorial não pode ser aplicada\n')
