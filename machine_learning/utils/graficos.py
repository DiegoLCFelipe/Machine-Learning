import seaborn as sns
import matplotlib.pyplot as plt

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
    plt.show()