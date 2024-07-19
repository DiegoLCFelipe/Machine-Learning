import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def estilo_visualizacao(function):
    def wrapper(*args, **kwargs):
        plt.style.use('seaborn-v0_8-dark-palette') 
        function(*args, **kwargs)  
    return wrapper

@estilo_visualizacao
def mapa_de_calor(dados):
    plt.figure(1)
    sns.heatmap(dados, cmap="Blues", annot=True)
    plt.title("Matriz de Correlações")
    plt.tight_layout()
    
    
def grafico_de_cargas(tabela_de_cargas, fator1, fator2):
    plt.figure(2)
    plt.scatter(x=tabela_de_cargas[fator1], y=tabela_de_cargas[fator2])
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.axhline(y=0, color='grey', ls='--')
    plt.axvline(x=0, color='grey', ls='--')
    plt.xlabel(str(fator1))
    plt.ylabel(str(fator2))

    for i, linha in tabela_de_cargas.iterrows():
        plt.gca().text(linha[fator1]+0.05, linha[fator2], i)
    plt.tight_layout()
    plt.title("Gráfico de Cargas")

def fatores_extraidos(tabelas_de_autovalores, variancia):
    plt.figure(3)
    plt.bar(x=tabelas_de_autovalores.index,
                height=tabelas_de_autovalores[variancia])
    
    count =0  
    for i, linha in tabelas_de_autovalores.iterrows():
        plt.gca().text(count-0.05, linha[variancia], round(linha[variancia], 3))
        count+=1
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticklabels([])
    plt.title("Fatores Extraidos")
    plt.tight_layout()

@estilo_visualizacao
def mapa_perceptual_anacor(dados_var1, dados_var2, per_inercia1, per_inercia2, label_var1, label_var2):
    sns.scatterplot(data=dados_var1, x=dados_var1[0], y=dados_var1[1])
    sns.scatterplot(data=dados_var2, x=dados_var2[0], y=dados_var2[1])
    plt.axhline(y=0, color='lightgrey', ls='--')
    plt.axvline(x=0, color='lightgrey', ls='--')
    plt.xlabel(f'{dados_var1.columns[0]}: {per_inercia1} da inércia', fontsize=8)
    plt.ylabel(f'{dados_var2.columns[0]}: {per_inercia2} da inércia', fontsize=8)
    plt.title("Mapa Perceptual", fontsize=12)

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'].iloc[0]+.02,
                    point['y'].iloc[0],
                    str(point['val'].iloc[0]))

    label_point(dados_var1[0],
                dados_var1[1],
                label_var1,
                plt.gca())
    
    label_point(dados_var2[0],
                dados_var2[1],
                label_var2,
                plt.gca())

def mostrar():
    plt.show()