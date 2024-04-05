data_x=input("introduza valores do wine\n")
data=data_x.split(";")
print(data)
fmap_data = map(float, data)
print(fmap_data)
flist_data = list(fmap_data)
print(flist_data)
data1 = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequalitywhite.csv",sep=";")
data2=data1.iloc[:0,:11]
data_preparation=pd.DataFrame([flist_data],columns=list(data2))
out=data2
for x in out:
print(x,data_preparation[x].values)
loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb'))
y_pred=loaded_model.predict(data_preparation)
print("wine quality",int(y_pred))