import pandas as pd
df=pd.read_csv('immigration.csv')
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
import seaborn as sb

import matplotlib.pyplot as plt


def max_pays():

    max=df.groupby('Country')['Total'].sum().sort_values(ascending=False)
    print(max.head(5))
    return

def visiualiation_par_continent():
    continent=df.groupby('Continent')['Total'].sum().sort_values(ascending=False)
    continent.plot(kind='bar')
    plt.ticklabel_format(style='plain', axis='y')
    plt.title('visiualiation_par_continent')
    plt.show()
    print(continent)

def croissance():
    try:
        country_v=input(str( 'Qu"elle pays voulez vous examiner  : '))
        country_v = country_v.capitalize()
        country=df.loc[df['Country']==country_v]
    except:
        print('ENTRER UN NOM DE PAYS VALIDE')
    x = str((input(('Qu" elle année voulez vous examiner 1980 A 2013 : '))))
    y=int(x)+1
    annne_N=country[x].to_string(index=False)
    annne_N1=country[str(y)].to_string(index=False)
    augmentation=int(annne_N1)/int(annne_N)
    for colone in country:
        if colone==x:
            print('Pays : ',country_v)
            print('Nombre d"immigre recense : ',annne_N)
            print('L"année suivante: ' ,annne_N1)
            if augmentation <0:
                print(f'Nous avons constaté une augmentation de {augmentation:.2f} %')
            else:
                print(f'Nous avons constaté une diminution de {abs(augmentation):.2f} %')

def visualisation_par_pays():
        years = list(map(str, range(1980, 2014)))
        LOL = df[['Country'] + years]

        country_v = input('Quel pays voulez-vous examiner : ').capitalize()
        country_data = LOL[LOL['Country'] == country_v]

        if country_data.empty:
            print("Le pays n'existe pas dans les données.")
            return

        country_data = country_data.set_index('Country').T
        country_data.plot(kind='line', legend=False)
        plt.title(f'Nombre d\'immigrés de {country_v} par année')
        plt.xlabel('Année')
        plt.ylabel('Nombre d\'immigrés')
        plt.ticklabel_format(style='plain', axis='y')
        plt.show()


def max_continent():
  continent=df.groupby('Continent')['Total'].sum().sort_values(ascending=False)
  continent_max_values=continent.max()
  continent_max_name=continent.idxmax()
  print('continent : ',continent_max_name)
  print('nombre d"imigré : ',continent_max_values)


def previlisuation ():
    print('les 5 premieres colones de donner')
    print(df.head(5))
    print('Nom columns')
    print(df.columns)
    print('information')
    print(df.info())
    print(df.describe().round())
def model():
    x=df[['2008','2009','2010','2011']]
    y=df["2012"]
    x_train,x_test,y_train_,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    model=LinearRegression()

    model.fit(x_train,y_train_)
    Y_predict=model.predict(x_test)
    sb.relplot(x=y_test,y=Y_predict)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()

    mse=mean_squared_error(y_test,Y_predict)
    r2=r2_score(y_test,Y_predict)
    validation=cross_val_score(model,x,y,cv=5,scoring='r2')


    print(mse)
    print(r2)
    print(validation)

def previsions_long_terme():
    df_yearly = df.groupby('Year')['Total'].sum()
    model = ARIMA(df_yearly, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    print(forecast)

def clustering_pays():
    years = list(map(str, range(1980, 2014)))
    X = df[years]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    plt.figure(figsize=(10, 6))
    sb.scatterplot(x='2000', y='2010', hue='Cluster', data=df, palette='Set1')
    plt.title('Clustering des Pays')
    plt.show()

def analyse_tendances_saisonnalite():
    df_yearly = df.groupby('Year')['Total'].sum()
    decomposition = sm.tsa.seasonal_decompose(df_yearly, model='multiplicative')
    fig = decomposition.plot()
    plt.show()

def detection_anomalies():
    years = list(map(str, range(1980, 2014)))
    X = df[years]
    iso_forest = IsolationForest(contamination=0.05)
    df['Anomalies'] = iso_forest.fit_predict(X)
    plt.figure(figsize=(10, 6))
    sb.scatterplot(x='2000', y='2010', hue='Anomalies', data=df, palette='Set1')
    plt.title('Détection des Anomalies')
    plt.show()