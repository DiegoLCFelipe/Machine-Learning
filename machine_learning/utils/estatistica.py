import pandas as pd
from typing import Series
from typing import DataFrame
from scipy.stats import chi2_contingency


def tabela_de_contingencia(variavel1: Series[float], variavel2: Series[float]) -> DataFrame[float]:
    """Função que gera a tabela de contingência - frequências com que cada combinação
      de categorias aparece em um conjunto de dados.

    Args:
        variavel1 (Series[float]): _description_
        variavel2 (Series[float]): _description_

    Returns:
        DataFrame[float]: Tabela de contingência
    """
    return pd.crosstab(variavel1, variavel2)

def chi2_tabela_de_contingecia(tabela_contingencia:DataFrame[float]) -> object: 
    """Analisa a significância estatística da associação entre as variáveis (teste qui²)

    Args:
        tabela_contingencia (DataFrame[float]): tabela contendo as fequências observadas

    Returns:
        object: contendo a estatística de teste, p-value e as frequências esperadas 
    """
    return chi2_contingency(tabela_contingencia)