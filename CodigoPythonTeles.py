"""
Como testar:
    python CodigoPythonTeles.py --modelo=1  # Regressao Logistica
    python CodigoPythonTeles.py --modelo=2  # Suporte Vector Machine
    python CodigoPythonTeles.py --modelo=3  # Arvore de Decisao
    python CodigoPythonTeles.py --modelo=3  # Random Forest
"""

# importar bibliotecas python
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score


def obter_modelo_preditivo(id_modelo):
    
    model = None
    nome = ''
    
    if id_modelo == '1': # regressao
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        nome = 'Regressão Logística'
    elif id_modelo == '2': # Suport Vector Machine
        print('passei aqui')
        from sklearn.svm import SVC
        model = SVC(kernel = 'linear', random_state = 1)
        nome = 'Suporte Vector Machine'
    elif id_modelo == '3':  # árvore de decisao
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion='entropy', random_state=0)
        nome = 'Árvore de Decisão'
    elif id_modelo == '4': # random forest
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        nome = 'Random Forest'
        
    return model, nome

def pre_treinamento(id_modelo):

    # importar dados
    base = pd.read_csv('case_Formatura.csv', encoding='ISO-8859-1')

    # selecionar variáveis previsoras
    previsores = base.iloc[:,[1, 2, 12, 17, 20, 21, 23, 27, 28]]

    # Identificar e remover alunos não formados (desistentes?)
    total_nao_formado = previsores.loc[previsores['SEMESTRES_CURSADOS'] < previsores['DURACAO_CURSO']].shape[0]

    if total_nao_formado > 0:
        previsores.drop(previsores.loc[previsores['SEMESTRES_CURSADOS'] < previsores['DURACAO_CURSO']].index, inplace=True)
        print('Foram removidos {} registros da base'.format(total_nao_formado))
    
    # tratar variáveis categóricas
    labelencoder_previsores = LabelEncoder()
    previsores.iloc[:, 0] = labelencoder_previsores.fit_transform(previsores.iloc[:, 0]) # unidade
    previsores.iloc[:, 1] = labelencoder_previsores.fit_transform(previsores.iloc[:, 1]) # turma
    previsores.iloc[:, 8] = labelencoder_previsores.fit_transform(previsores.iloc[:, 8]) # possui fies

    # Definir e tratar classes de saída (semestres para conclusão do curso)
    classe = pd.DataFrame(previsores['SEMESTRES_CURSADOS'] - previsores['DURACAO_CURSO'], columns=['SEMESTRES_ATRASADOS'])
    total_classe = classe['SEMESTRES_ATRASADOS'].unique().shape[0] 

    # escalonamento
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)

    return previsores, classe

def treinamento(previsores, classe, classificador):

    # segmentação: treino e teste
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.10, random_state=0)

    # Treinamento
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)

    return previsoes, classe_teste

    

def _main():

    # tratar parâmetro de entrada: definir modelo de classificação
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', help='1=Regressao, 2=SVM, 3=Arvore de Decisao e 4=Random Forest', default=1)
    args = parser.parse_args()
   
    classificador, nome_classificador = obter_modelo_preditivo(args.modelo)

    # etapa 1: Pré-treinamento
    previsores, classe = pre_treinamento(args.modelo)

    # etapa 2: Treinamento
    previsoes, classe_teste = treinamento(previsores, classe, classificador)

    # etapa 3: Saída
    precisao = accuracy_score(classe_teste, previsoes)
    matriz = confusion_matrix(classe_teste, previsoes)

    print('----------------------------------------------------------------' )
    print('Modelo: {} >> Precisão: {}'.format(nome_classificador, precisao))
    print('----------------------------------------------------------------' )
    print('')
    print('Fim')
  

if __name__ == '__main__':
    _main()





