
# coding: utf-8

# <h3>Análise Inicial dos Dados</h3>
# <h4>Variáveis Preditoras:</h4>
# <pre>
# Índice     Nome                 Tipo
# .................................................
#    1       COD_UNIDADE          Categórica-Nominal
#    2       COD_TURMA            Categórica-Nominal
#   12       DURACAO_CURSO        Numérica-Discreta
#   17       NR_TOTAL_DISCIPLINAS Numérica-Discreta
#   20       T_IDADE              Numérica-Discreta
#   21       NOTA                 Numérica-Contínua
#   23       POSSUI_ENEM          Numérica-Discreta 
#   27       SEMESTRES_CURSADOS   Numérica-Discreta
#   28       POSSUI FIES          Categórica-Ordinal
# </pre>
# 
# <h4>Classes:</h4>
# <pre>
# Nome                 Tipo
# ......................................
# SEMESTRES_ATRASADOS  Numérica-Discreta
# 
# 
# Foram identificadas 29 classes: 
#     [0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 
#     15 16 18 20 21 22 24 40 57 62 64 80 90 98]
# </pre>

# In[1]:


# importar bibliotecas python
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[2]:
#modelo = 'regressao' # precisão = 0.8460532054077627
#modelo = 'svn'       # precisão = 0.9994185201337403
#modelo = 'ad'       # Precisão = 0.9997092600668702
modelo = 'rf'       # Precisão = 0.9985463003343509

if modelo == 'regressao':
    from sklearn.linear_model import LogisticRegression
    classificador = LogisticRegression()
elif modelo == 'svn':
    from sklearn.svm import SVC
    classificador = SVC(kernel = 'linear', random_state = 1)
elif modelo == 'ad':  # árvore de decisao
    from sklearn.tree import DecisionTreeClassifier
    classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
elif modelo == 'rf': # random forest
    from sklearn.ensemble import RandomForestClassifier
    classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        
# importar dados
base = pd.read_csv('case_Formatura.csv', encoding='ISO-8859-1')


# In[3]:


# Definir e tratar variáveis preditoras
previsores = base.iloc[:,[1, 2, 12, 17, 20, 21, 23, 27, 28]]

# Identificar e remover alunos não formados
total_nao_formado = previsores.loc[previsores['SEMESTRES_CURSADOS'] < previsores['DURACAO_CURSO']].shape[0]

if total_nao_formado > 0:
    previsores.drop(previsores.loc[previsores['SEMESTRES_CURSADOS'] < previsores['DURACAO_CURSO']].index, inplace=True)
    print('Foram removidos {} registros da base'.format(total_nao_formado))
else:
    print('Nenhum registro foi removido')


# In[4]:


labelencoder_previsores = LabelEncoder()
previsores.iloc[:, 0] = labelencoder_previsores.fit_transform(previsores.iloc[:, 0]) # unidade
previsores.iloc[:, 1] = labelencoder_previsores.fit_transform(previsores.iloc[:, 1]) # turma
previsores.iloc[:, 8] = labelencoder_previsores.fit_transform(previsores.iloc[:, 8]) # possui fies


# In[5]:


# Definir e tratar classes (semestres para conclusão do curso)
classe = pd.DataFrame(previsores['SEMESTRES_CURSADOS'] - previsores['DURACAO_CURSO'], columns=['SEMESTRES_ATRASADOS'])
total_classe = classe['SEMESTRES_ATRASADOS'].unique().shape[0] 
# print(classes.describe())
# print('Total de classes: ', total_classe)
# print('Lista de classes: ', np.sort(classe['SEMESTRES_ATRASADOS'].unique()))


# In[13]:


# escalonamento
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)


# In[21]:


# segmentação: treino e teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.10, random_state=0)


# In[22]:

# Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# In[24]:


# matriz de confusao
 from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)


# In[25]:


import collections
collections.Counter(classe_teste)


# In[26]:


print('Precisão =', precisao)
print('Fim')


# In[ ]:




