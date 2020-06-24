
import pandas as pd
from pandas.plotting._matplotlib import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv(
    "C:/Users/Devna Chaturvedi/Desktop/Summer Semester - Python/Python lesson 5/Python_Lesson5/winequality-red.csv")

X = train_df.drop('quality', axis=1)
y = train_df['quality']



# Train and Test splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
# Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# Statistical characteristics of each numerical feature
print(train_df.describe())
# Histograms
train_df.hist(bins=10,figsize=(6, 5))
plt.show()
# Density
train_df.plot(kind='density', subplots=True, layout=(4,3), sharex=False)
plt.show()


# Create pivot_table
colum_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
df_pivot_table = train_df.pivot_table(colum_names,
               ['quality'], aggfunc='median')
print(df_pivot_table)
corr_matrix = train_df.corr()
print(corr_matrix["quality"].sort_values(ascending=False))

colum_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
# Correlation matrix
correlations = train_df.corr()
# Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
# Generate Color Map
colormap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate Heat Map, allow annotations and place floats in map
sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
ax.set_xticklabels(
    colum_names,
    rotation=45,
    horizontalalignment='right'
);
ax.set_yticklabels(colum_names);
plt.show()
# Scatterplot Matrix
sm = scatter_matrix(train_df, figsize=(6, 6), diagonal='kde')
#Change label rotation
[s.xaxis.label.set_rotation(40) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.6,0.5) for s in sm.reshape(-1)]
#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LogisticRegression', LogisticRegression()))
# evaluate the model
results = []
names = []
scoring = 'accuracy'
for name, model in models:
   kfold = model_selection.KFold(n_splits=10, random_state=seed)
   cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print(train_df.describe())
train_df['quality'].unique()
print(train_df['quality'].unique())
train_df.quality.value_counts().sort_index()
correlations = train_df.corr()['quality'].drop('quality')
print(correlations)

correlations.plot(kind='bar')
plt.show()

print(abs(correlations)>0.2)
