import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity


def matriz_de_correlacoes(dados):
    return dados.corr()

def teste_de_esfericidade_de_bartlett(dados, significancia):
    bartlett, p_valor = calculate_bartlett_sphericity(dados)
    ordem_da_matriz = dados.shape[1]
    graus_de_liberdade = (ordem_da_matriz * (ordem_da_matriz - 1)) // 2
    return bartlett, p_valor, graus_de_liberdade
    
class AnaliseFatorial:
    """
    Classe para realizar Análise Fatorial em um conjunto de dados.

    Métodos
    -------
    __init__(dados, numero_de_fatores):
        Inicializa a instância da classe com os dados e o número de fatores.
    
    calcula_autovalores():
        Calcula e retorna os autovalores da análise fatorial.
    
    calcula_cargas_fatoriais():
        Calcula e retorna as cargas fatoriais.
    
    calcula_comunalidades():
        Calcula e retorna as comunalidades.
    
    extrai_fatores_para_as_observacoes():
        Extrai e retorna os fatores para cada observação.
    
    aplicar_criterio_de_kaiser():
        Aplica o critério de Kaiser e retorna o número de fatores com autovalores maiores ou iguais a 1.
    
    mostra_tabela_de_autovalores():
        Calcula e exibe a tabela de autovalores, variâncias e variâncias acumuladas.
    
    mostra_tabela_de_cargas():
        Calcula e exibe a tabela de cargas fatoriais.
    
    mostra_tabela_de_comunalidades():
        Calcula e exibe a tabela de comunalidades.
    """

    def __init__(self, dados, numero_de_fatores):
        """
        Inicializa a instância da classe AnaliseFatorial.

        Parâmetros
        ----------
        dados : DataFrame
            O conjunto de dados a ser analisado.
        numero_de_fatores : int
            O número de fatores a ser extraído na análise fatorial.
        """
        self.dados = dados
        self.numero_de_fatores = numero_de_fatores
        self.analise_fatorial = self._analise_fatorial()

    def _analise_fatorial(self):
        """
        Realiza a análise fatorial usando o método principal.

        Retorna
        -------
        FactorAnalyzer
            O objeto FactorAnalyzer ajustado aos dados.
        """
        return FactorAnalyzer(n_factors=self.numero_de_fatores, method='principal', rotation=None).fit(self.dados)

    def get_autovalores(self):
        """
        Calcula e retorna os autovalores da análise fatorial.

        Retorna
        -------
        list
            Uma lista de autovalores, variância explicada e variância acumulada.
        """
        return self.analise_fatorial.get_factor_variance()

    def get_cargas_fatoriais(self):
        """
        Calcula e retorna as cargas fatoriais.

        Retorna
        -------
        DataFrame
            Um DataFrame contendo as cargas fatoriais.
        """
        return self.analise_fatorial.loadings_

    def get_comunalidades(self):
        """
        Calcula e retorna as comunalidades.

        Retorna
        -------
        list
            Uma lista contendo as comunalidades.
        """
        return self.analise_fatorial.get_communalities()

    def get_fatores(self):
        """
        Extrai e retorna os fatores para cada observação.

        Retorna
        -------
        DataFrame
            Um DataFrame contendo os fatores para cada observação.
        """
        fatores = pd.DataFrame(self.analise_fatorial.transform(self.dados))
        fatores.columns = [f"Fator {i+1}" for i in range(fatores.shape[1])]
        return fatores
    
    def get_scores(self):
        """
        Calcula e retorna scores fatoriais.

        Retorna
        -------
        list
            Uma lista contendo scores fatoriais.
        """
        return self.analise_fatorial.weights_

    def criterio_de_kaiser(self):
        """
        Aplica o critério de Kaiser para determinar o número de fatores com autovalores maiores ou iguais a 1.

        Retorna
        -------
        int
            O número de fatores com autovalores maiores ou iguais a 1.
        """
        autovalores = self.get_autovalores()[0]
        return len([i for i in autovalores if i >= 1])
    
    def ranking(self):
        ranking = 0
        ranking_list = []
        autovalores = self.tabela_de_autovalores()
        for index, item in enumerate(list(autovalores.index)):
            variancia = autovalores.loc[item]['Variância']
            ranking = ranking + self.dados_com_fatores()[autovalores.index[index]]*variancia
            ranking_list.append(ranking)

        return pd.Series(ranking_list[0] + ranking_list[1], name='Rank')

    def tabela_de_autovalores(self):
        """
        Calcula e exibe a tabela de autovalores, variâncias e variâncias acumuladas.
        """
        autovalores_dos_fatores = self.get_autovalores()
        tabela_autovalores = pd.DataFrame(autovalores_dos_fatores)
        tabela_autovalores.columns = [f"Fator {i+1}" for i in range(tabela_autovalores.shape[1])]
        tabela_autovalores.index = ['Autovalor', 'Variância', 'Variância Acumulada']
        tabela_autovalores = tabela_autovalores.T
        return tabela_autovalores

    def tabela_de_cargas(self):
        """
        Calcula e exibe a tabela de cargas fatoriais.
        """
        cargas_fatoriais = pd.DataFrame(self.get_cargas_fatoriais())
        cargas_fatoriais.columns = [f"Fator {i+1}" for i in range(cargas_fatoriais.shape[1])]
        cargas_fatoriais.index = self.dados.columns
        return cargas_fatoriais

    def tabela_de_comunalidades(self):
        """
        Calcula e exibe a tabela de comunalidades.
        """
        comunalidades = pd.DataFrame(self.get_comunalidades(), columns=['Comunalidades'])
        comunalidades.index = self.dados.columns
        return comunalidades
    
    def tabela_de_scores(self):
        """
        Calcula e exibe a tabela de scores fatoriais.
        """
        scores_fatoriais = pd.DataFrame(self.get_scores())
        scores_fatoriais.columns = [f"Fator {i+1}" for i, v in enumerate(scores_fatoriais.columns)]
        scores_fatoriais.index = self.dados.columns
        return scores_fatoriais
    
    def dados_com_fatores(self):
        return pd.concat([self.dados.reset_index(drop=True),
                          self.get_fatores()], axis=1)
    

    def dados_rankeados(self):
        return pd.concat([self.dados_com_fatores(), self.ranking()], axis=1)