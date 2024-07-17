import sys, os
import yaml
import pandas as pd

os.system('cls')
sys.path.insert(0,os.path.abspath(os.curdir))
from machine_learning.models.analise_correspondencia_simples import AnaliseCorrespondenciaSimples
from machine_learning.utils.logger import LogHandler
import machine_learning.utils.graficos as visualizacao

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
with open(CURRENT_PATH + '/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

PATH = config['path']
VARIAVEIS = config['variaveis']
dados = pd.read_excel(PATH, decimal=',')
dados_analise = dados[VARIAVEIS]

print(pd.crosstab(dados_analise[VARIAVEIS[0]], dados_analise[VARIAVEIS[1]]))
#analise =  AnaliseCorrespondenciaSimples(dados_analise)
#print(analise.get_autovalores)