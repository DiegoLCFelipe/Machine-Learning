import sys, os
import yaml
import pandas as pd


os.system('cls')
sys.path.insert(0,os.path.abspath(os.curdir))
from machine_learning.models.analise_correspondencia_simples import AnaliseCorrespondenciaSimples
from machine_learning.utils.logger import LogHandler
import machine_learning.utils.graficos as visualizacao
from machine_learning.utils.estatistica import AnaliseDeFrequencias

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
with open(CURRENT_PATH + '/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

PATH = config['path']
VARIAVEIS = config['variaveis']
SIGNIFICANCIA = config['significancia']
dados = pd.read_excel(PATH, decimal=',')
dados_analise = dados[VARIAVEIS]

log = LogHandler()

analise_de_frequencias = AnaliseDeFrequencias(dados_analise[VARIAVEIS[0]], dados_analise[VARIAVEIS[1]])
analise_correspondencia = AnaliseCorrespondenciaSimples(analise_de_frequencias.get_tabela_de_frequencias())

estatistica, p_valor, graus_de_liberdade, _ = analise_de_frequencias.tabela_chi2()

log.log_info_timestamp(f'Teste de Chi-Quadrado\n\n'
                       f'Chi-quadrado: {estatistica:.3f}\n'
                       f'P-valor: {p_valor:.2f}\n'
                       f'Graus de liberdade: {graus_de_liberdade:.1f}\n')

if p_valor > SIGNIFICANCIA:
    log.log_warning(f'Aceita-se H0: a associação das variáveis não se dá de forma aleatória.')
    
else:
    log.log_info(f'Rejeita-se H0: as variáveis se associam de forma aleatória.')
    analise_de_frequencias.tabela_de_frequencias_absolutas_esperadas()
    analise_de_frequencias.tabela_chi2()

    tabela_de_residuos = analise_de_frequencias.tabela_de_residuos()
    tabela_de_residuos_ajustados = analise_de_frequencias.tabela_de_residuos_ajustados()
    tabela_de_residuos_padronizados_ajustados = analise_de_frequencias.tabela_de_residuos_padronizados_ajustados()

    perc_variancia_var1 = analise_correspondencia.autovalores().iloc[0,1]
    perc_variancia_var2 = analise_correspondencia.autovalores().iloc[1,1]
    x_coord_grafico_perceptual = analise_correspondencia.coordenadas_linhas()
    y_coord_grafico_perceptual = analise_correspondencia.coordenadas_colunas()
    label_var1 = x_coord_grafico_perceptual[[x_coord_grafico_perceptual.columns[0]]]
    label_var2 = y_coord_grafico_perceptual[[y_coord_grafico_perceptual.columns[0]]]


    with open (CURRENT_PATH + '/log/residuos.log', 'w', encoding='utf-8') as file:
        file.write('Tabela de Resíduos\n')
        file.write(tabela_de_residuos.to_string())
        file.write('\nTabela de Resíduos Ajustados\n')
        file.write(tabela_de_residuos_ajustados.to_string())
        file.write('\nTabela de Resíduos Padronizados Ajustados\n')
        file.write(tabela_de_residuos_padronizados_ajustados.to_string())  

visualizacao.mapa_perceptual_anacor(
        x_coord_grafico_perceptual,
        y_coord_grafico_perceptual,
        perc_variancia_var1,
        perc_variancia_var2,
        label_var1,
        label_var2
    )
visualizacao.mostrar()



   


