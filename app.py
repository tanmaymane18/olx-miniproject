import streamlit as st
import io
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np
from getAttrs import getAttrs


@st.cache
def findBest(trainSet):
    min_samples = range(2,20)
    eps = np.arange(0.01,1.01, 0.01)

    silScores = []
    append = silScores.append

    for minSamples in min_samples:
        for ep in eps:
            labels = DBSCAN(min_samples=minSamples, eps=ep).fit(trainSet).labels_
            try:
                score = silhouette_score(trainSet, labels)
                params = (minSamples, ep)
                append((score, params))
            except:
                pass

    sortedScores = sorted(silScores, key=lambda x:x[0])
    score, params = sortedScores[-1]

    return score, params[0], params[1]



df = pd.read_csv("./bikes.csv")
#make = st.selectbox("Select The Make: " ,df['make'].unique())
#model = st.selectbox("Select The Make: " ,df[df['make'] == make]['model'].unique())

url = st.text_input("Url")

make, model, year, running, price = getAttrs(url)

st.write(make, model, year, price)

train = df[(df['model'] == model) & (df['make'] == make)][['running', 'price', 'year', 'make', 'model']]
train['price'] = train['price'].apply(lambda x: int(x.replace(',', '').split()[1]))
train = pd.concat([train, pd.DataFrame(data=[[running, price, year, make, model]], columns=['running', 'price', 'year', 'make', 'model'])])
st.write(train)

std = StandardScaler()
featureTrain = train[['price', 'running', 'year']]
stdTrain = std.fit_transform(featureTrain)

#min_samples = st.slider(label='min_samples', min_value=2, max_value=20)
#eps = st.slider(label='eps', min_value=0.1, max_value=1.0)

#price, year, kms = getAttrs("")

score, min_samples, eps = findBest(stdTrain)
dbscan = DBSCAN(min_samples=min_samples, eps=eps).fit(stdTrain)
labels = dbscan.labels_
st.write("Silhouette Socre: ", score)
if labels[-1] == -1:
    st.warning("The Given Ad is an Outlier")
else:
    st.success("The Given Ad is not an Outlier")
fig = px.scatter_3d(featureTrain, x='price', y='year', z='running',
              color=labels)
st.plotly_chart(fig)
