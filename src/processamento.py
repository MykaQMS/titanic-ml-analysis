import re
import pandas as pd
import numpy as np

def extrair_titulo(nome: str) -> str:
    """
    Extrai o título de cortesia (Mr, Mrs, Miss, Master, Raro) do nome do passageiro.
    Exemplo: "Braund, Mr. Owen Harris" -> "Mr"
    """
    if not isinstance(nome, str):
        return "Raro"
    
    match = re.search(r' ([A-Za-z]+)\.', nome)
    if not match:
        return "Raro"
    
    titulo = match.group(1)
    
    # Agrupamos títulos raros ou equivalentes para simplificar a análise
    titulos_raros = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    
    if titulo in titulos_raros:
        return 'Raro'
    elif titulo in ['Mlle', 'Ms']:
        return 'Miss'
    elif titulo == 'Mme':
        return 'Mrs'
    elif titulo in ['Mr', 'Mrs', 'Miss', 'Master']:
        return titulo
    else:
        return 'Raro'

def preparar_dados(df_treino: pd.DataFrame, df_teste: pd.DataFrame):
    """
    Realiza o pré-processamento e a engenharia de atributos nos dados do Titanic:
    1. Junta temporariamente as bases para aplicar as mesmas transformações.
    2. Extrai os Títulos (Mr, Mrs, Miss, Master).
    3. Preenche idades nulas com a mediana do seu respectivo Título.
    4. Preenche valores faltantes de Tarifa e Embarque.
    5. Cria as variáveis TamanhoFamilia e Sozinho.
    6. Converte colunas de texto em números usando One-Hot Encoding (pd.get_dummies).
    """
    # Guardamos os IDs do teste para gerar o arquivo de submissão depois
    ids_teste = df_teste['PassengerId'].copy()
    
    # Marcamos quem pertence ao treino e ao teste
    df_treino_copy = df_treino.copy()
    df_teste_copy = df_teste.copy()
    
    df_treino_copy['eh_treino'] = 1
    df_teste_copy['eh_treino'] = 0
    df_teste_copy['Survived'] = np.nan
    
    # Concatenamos para tratar tudo junto
    dados = pd.concat([df_treino_copy, df_teste_copy], sort=False).reset_index(drop=True)
    
    # 1. Extração do Título
    dados['Titulo'] = dados['Name'].apply(extrair_titulo)
    
    # 2. Preenchimento de Idades Nulas (mediana por Título)
    medianas_idade = dados.groupby('Titulo')['Age'].median()
    def preencher_idade(linha):
        if pd.isna(linha['Age']):
            return medianas_idade.get(linha['Titulo'], dados['Age'].median())
        return linha['Age']
        
    dados['Age'] = dados.apply(preencher_idade, axis=1)
    
    # 3. Preenchimento de Tarifa e Embarque
    dados['Fare'] = dados['Fare'].fillna(dados['Fare'].median())
    dados['Embarked'] = dados['Embarked'].fillna(dados['Embarked'].mode()[0])
    
    # 4. Criando novas variáveis de família
    dados['TamanhoFamilia'] = dados['SibSp'] + dados['Parch'] + 1
    dados['Sozinho'] = (dados['TamanhoFamilia'] == 1).astype(int)
    
    # 5. Seleção de atributos principais para o modelo
    colunas_relevantes = [
        'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 
        'Titulo', 'TamanhoFamilia', 'Sozinho'
    ]
    
    # Aplicando One-Hot Encoding (Dummies) nas colunas de texto
    dados_dummies = pd.get_dummies(dados[colunas_relevantes], drop_first=True)
    
    # Recolocando a coluna alvo (Survived) e a marcação de treino
    dados_dummies['Survived'] = dados['Survived']
    dados_dummies['eh_treino'] = dados['eh_treino']
    
    # Separando de volta em treino e teste
    treino_processado = dados_dummies[dados_dummies['eh_treino'] == 1].drop(columns=['eh_treino'])
    teste_processado = dados_dummies[dados_dummies['eh_treino'] == 0].drop(columns=['eh_treino', 'Survived'])
    
    X_treino = treino_processado.drop(columns=['Survived'])
    y_treino = treino_processado['Survived'].astype(int)
    X_teste = teste_processado
    
    return X_treino, y_treino, X_teste, ids_teste
