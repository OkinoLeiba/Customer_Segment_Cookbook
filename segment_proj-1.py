# %% [markdown]
# # <span style="background:rgba(185, 155, 0, 0.5); color:rgb(210,105,3); outline-style:solid; outline-color:rgb(90, 90, 90); border-radius: 25px; letter-spacing: 18px;"><b>Customer Segmentation</b></span>

# %% [markdown]
# <a href="https://www.linkedin.com/in/okinoleiba" style="">Okino Kamali Leiba</a>

# %% [markdown]
# ### <span style="border-style:dotted; border-color:rgb(255,0,255); letter-spacing: 30px; color:rgb(205,92,92);">Parameters</span>
# <span style="color:rgb(0,255,0); text-shadow: rgb(50,205,50);">
# <ul>
# <li>Use case: Identify market segment from a larger population.</li>
# <li>Data: Really fake data from our really fake mall.</li>
# <li>Tools: Jupyter Notebook</li>
# <li>Cross-instrument: none</li>
# <li>Documentation: none</li>
# </ul>
# </span>

# %% [markdown]
# ### <span style="background-image:conic-gradient(rgba(185, 155, 0, 0.5), rgba(135, 93, 71,0.5), green, rgba(112, 225, 255,0.5), rgba(255, 238, 150,0.3)); color:rgb(139,0,139); letter-spacing:14px;">Table of Content</span>
# <span style="color:rgb(50,205,50);">
# <ul>
# <li>Introduction</li> 
# <li>Exploratory Data Analysis</li>
# <li>Create Segments and Clusters <small>(KMeans Clustering Algorithm)</small></li>
# <li>Final Analysis</li>
# </ul>
# </span>

# %% [markdown]
# ### <span style="text-shadow:0 0 3px #FFFF00, 0 0 5px #0000FF; letter-spacing:14px; color:red;">Introduction</span>

# %% [markdown]
# <span style="color:rgb(255,215,0); box-shadow: 5px 5px rgba(185, 155, 0, 0.5);">Imports</span>
# 
# <span style="color: rgb(60,179,113);">
# <ul>
# <li>SYS module for access to variables and functions used or maintained by the interpreter</li>
# <li>Pandas for data analysis and manipulation</li>
# <li>Numpy for general array computations</li>
# <li>Matplotlib for creating static, animated, and interactive visualizations in Python</li>
# <li>Seaborn for Python data visualization based on Matplotlib</li>
# <li>Scikit-Learn for machine learning library that supports supervised and unsupervised learning</li>
# </ul>
# </span>
# 

# %%
import sys, pandas as pd, numpy as np, matplotlib as plt, seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
ImportWarning.__suppress_context__

# %%
file_path="/Users/Owner/source/vsc_repo/customer_segment_cookbook/mall_customers.csv"
segment_data = pd.read_csv(file_path, delimiter=",", header=0, nrows=300, na_values="NaN", keep_default_na=True, skip_blank_lines=False, 
encoding="utf-8")
segment_data.columns

# %% [markdown]
# ### <span style="background-image:repeating-radial-gradient(rgba(185, 155, 0, 0.5), rgba(135, 93, 71,0.5) 20%, green, rgba(112, 225, 255,0.5), rgba(255, 238, 150,0.3) 20%); color:rgb(173,255,47); letter-spacing:5px;">Exploratory Data Analysis</span>

# %%
segment_data.head(5)

# %%
segment_data.tail(5)

# %%
segment_data.info

# %%
if all(segment_data.isna()) or all(segment_data.isnull()): 
    print("All Good") 
else: 
    print("Danger, Will Robinson")

# %%
segment_data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].describe().round(decimals=3)

# %%
count_data = segment_data['Spending Score (1-100)'].value_counts(normalize=True, sort=True, ascending=True).round(decimals=3)
count_data

# columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
# count_data = []
# for iterable in columns:
#     count_data = segment_data[iterable].value_counts(normalize=True, sort=True, ascending=True).round(decimals=3)
#     # print(count_data[iterable])
# count_data


# %% [markdown]
# <span style="box-shadow: 5px 5px rgb(119,136,153) inset; color:rgb(0,0,205); padding-left:5px;"><b>Univariate Analysis and Visualization</b></span>

# %%
sns.distplot(segment_data["Annual Income (k$)"], color="purple",hist=True);

# %%
sns.distplot(segment_data["Spending Score (1-100)"], kde=True, color="purple");

# %%

fig, axe = plt.pyplot.subplots(nrows=1,ncols=3, constrained_layout=True, figsize=(10, 4))

sns.distplot(segment_data["Spending Score (1-100)"], kde=True, color="purple", ax=axe[0]);
sns.distplot(segment_data["Annual Income (k$)"], color="silver",hist=True, ax=axe[1]);
sns.distplot(segment_data["Age"], color="purple",hist=True, ax=axe[2]);

# fig = plt.pyplotfigure(constrained_layout=True, figsize=(10, 4))
# subfigs = fig.subfigures(1, 2, wspace=0.07)
# sns.distplot(segment_data["Spending Score (1-100)"], kde=True, color="silver", ax=subfigs[0]);
# sns.distplot(segment_data["Annual Income (k$)"], color="orange",hist=True, ax=subfigs[1]);
# sns.distplot(segment_data["Age"], color="orange",hist=True, ax=subfigs[2]);

# %%
columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for iterable in columns:   
    fig = plt.pyplot.figure(figsize=(6, 3))
    # fig, axe = plt.pyplot.subplots(nrows=1,ncols=3)
    # fig, axe = plt.pyplot.subplots(nrows=1,ncols=3, constrained_layout=True, figsize=(10, 4))
    sns.distplot(segment_data[iterable], color="purple",hist=True);
    
   

# %%
sns.kdeplot(segment_data["Spending Score (1-100)"], shade=True, color="purple");

# %%
import matplotlib.image as mpimg
sns.kdeplot(segment_data["Spending Score (1-100)"], shade=True);
img = mpimg.imread("/Users/Owner/Pictures/Black Panther_F.jpg")
img = img[:, :, 0]
imgplot = plt.pyplot.imshow(img)


# %%
columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for iterable in columns:   
    fig = plt.pyplot.figure(figsize=(6, 3))
    #fig, axs = plt.pyplot.subplot()
    sns.boxplot(data=segment_data, x=segment_data["Gender"],  y=segment_data[iterable],color="purple");

# %%
sns.scatterplot(data=segment_data, x=segment_data["Annual Income (k$)"], y=segment_data["Spending Score (1-100)"]);

# %%
sns.violinplot(data=segment_data, x="Annual Income (k$)", y="Gender", palette="Spectral")
sns.despine()

# %%

drop_data = segment_data.drop("CustomerID", axis=1)
sns.pairplot(data=drop_data, hue="Gender", kind="scatter", dropna=True);

# %%
mean_data = segment_data.groupby("Gender")['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean().round(decimals=3)
mean_data

# %%
drop_data = segment_data.drop("CustomerID", axis=1)
drop_data.corr().round(decimals=3)

# %%
sns.heatmap(drop_data.corr(),cmap="coolwarm")

# %% [markdown]
# ### <span style="background-image:repeating-linear-gradient(to right, rgba(185, 155, 0, 0.5), rgba(135, 93, 71,0.5), green, rgba(112, 225, 255,0.5), rgba(255, 238, 150,0.3)); color:black;">Create Segments and Clusters <small>(KMeans Clustering Algorithm)</small></span>

# %% [markdown]
# <span style="box-shadow: 5px 5px rgb(107,142,35), 10px 10px rgb(175,238,238), 5px 5px rgb(218,112,214); color:orange;"><strong>Clustering - Univariate, Bivariate, and Multivariate</strong></span>

# %% [markdown]
# <span style="box-shadow:5px 5px rgb(218,112,214); color:rgb(210,180,140);"><small>Univariate Clustering</small></span>

# %%
from sklearn.cluster import KMeans

# %%
uni_cluster = KMeans(init="k-means++", n_init=3, n_clusters=3)

# segment_data["Annual Income (k$)"].array.reshape(1, -1)
# uni_cluster = sklearn.cluster.k_means(segment_data["Annual Income (k$)"], n_clusters=8)

# %%
uni_cluster.fit(segment_data[["Annual Income (k$)"]])
uni_cluster.labels_
segment_data["Income Cluster"] = uni_cluster.labels_
uni_cluster_data = segment_data
uni_cluster_data.head(0)

# %%
# https://scikit-learn.org/stable/modules/clustering.html#k-means
uni_cluster.inertia_

# %%
uni_cluster.cluster_centers_
# best_fit = uni_cluster.fit_predict(segment_data[['Age', 'Annual Income (k$)','Spending Score (1-100)']])


# %%
inertia_scores = []
for iterable in range(1,11):
        uni_kmeans = KMeans(n_clusters=iterable ).fit(uni_cluster_data[["Annual Income (k$)"]])
        inertia_scores.append(uni_kmeans.inertia_)
# inertia_scores = [iterable.fit(segment_data[["Annual Income (k$)"]]) for iterable in range(1,11) lambda kmeans: kmeans = sklearn.cluster.KMeans(n_cluster=iterable) ]


# %%
plt.pyplot.plot(range(1,11),inertia_scores)
plt.pyplot.scatter(range(1,11),inertia_scores)
plt.pyplot.plot(3, inertia_scores[3], marker="x", color="green", linestyle="dashed", linewidth=2, markersize=12)
plt.pyplot.xlabel("Number of Clusters", size=13)
plt.pyplot.ylabel("Inertia Value", size=13)
plt.pyplot.title("Different Inertia Values for Different Number of Clusters", size=17);


# %% [markdown]
# <span style="color: rgb(139, 0, 0);">Univariate Analysis</span>

# %%
uni_mean = segment_data.groupby("Income Cluster")['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean().round(decimals=3)
uni_mean

# %%
uni_count = uni_cluster_data["Income Cluster"].value_counts(normalize=True, sort=True, ascending=True).round(decimals=3)
uni_count

# %% [markdown]
# <span style="box-shadow: 5px 5px rgb(154,205,50), 5px 5px rgb(64,224,208) inset; color:rgb(255,0,0); padding-left: 5px;">Bivariate Clustering</span>

# %%
from sklearn.cluster import KMeans

# %%
bi_cluster = KMeans(init="k-means++", n_init=3, n_clusters=5)

# %%
bi_cluster.fit(segment_data[["Annual Income (k$)", "Spending Score (1-100)"]])
bi_cluster.labels_
segment_data["Income and Spending Cluster"] = bi_cluster.labels_
bi_cluster_data = segment_data
bi_cluster_data.head(0)

# %%
# https://scikit-learn.org/stable/modules/clustering.html#k-means
bi_cluster.inertia_

# %%
bi_cluster.cluster_centers_

# %%
bi_inertia_scores = []
for iterable in range(1,11):
        bi_kmeans = KMeans(n_clusters=iterable ).fit(bi_cluster_data[["Annual Income (k$)", "Spending Score (1-100)" ]])
        bi_inertia_scores.append(bi_kmeans.inertia_)


# %%
plt.pyplot.plot(range(1,11),bi_inertia_scores)
plt.pyplot.scatter(range(1,11),bi_inertia_scores, edgecolors="black")
plt.pyplot.plot(5, bi_inertia_scores[5], marker="x", color="green", linestyle="dashed", linewidth=2, markersize=12)
plt.pyplot.xlabel("Number of Clusters", size=13)
plt.pyplot.ylabel("Inertia Value", size=13)
plt.pyplot.title("Different Inertia Values for Different Number of Clusters", size=17);


# %%
cluster_centriod = pd.DataFrame(bi_cluster.cluster_centers_)
cluster_centriod.columns = ["x","y"]
plt.pyplot.figure(figsize=(10,8))
plt.pyplot.scatter(x=cluster_centriod["x"], y=cluster_centriod["y"], s=300, color="black", marker="p")
sns.scatterplot(data=bi_cluster_data, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Income and Spending Cluster", palette="husl");

# %% [markdown]
# <span style="color:rgb(184,134,11);">Bivariate Analysis</span>

# %%
pd.crosstab(index=segment_data["Income and Spending Cluster"], columns=segment_data["Gender"], dropna=True, normalize="all")

# %%
pd.crosstab(index=segment_data["Income and Spending Cluster"], columns=segment_data["Age"], dropna=True, normalize="all")

# %%
bi_mean = segment_data.groupby("Income and Spending Cluster")['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean().round(decimals=3)
bi_mean

# %%
bi_count = bi_cluster_data["Income and Spending Cluster"].value_counts(normalize=True,sort=True,ascending=True).round(decimals=3)
bi_count


# %% [markdown]
# <span style="box-shadow:5px 5px rgb(0, 191, 255); color:rgb(210,180,140);">Multivariate Clustering</span>

# %%
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


# %%
encode_cat = pd.get_dummies(segment_data, drop_first=True)
encode_cat.head()


# %%
standard_cat = encode_cat[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender_Male"]]
standard_cat.head(5)

# %%
standardized_data = pd.DataFrame(scale.fit_transform(standard_cat))
standardized_data = standardized_data.set_axis(["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender_Male"], axis=1)
standardized_data.head(5)


# %%
from sklearn.cluster import KMeans

# %%
multi_cluster = KMeans(init="k-means++", n_init=3, n_clusters=4)

# %%
multi_cluster.fit(standardized_data[["Annual Income (k$)", "Spending Score (1-100)"]])
multi_cluster.labels_
standardized_data["Income and Spending Cluster"] = multi_cluster.labels_
multi_cluster_data = standardized_data
multi_cluster_data.head(5)

# %%
# https://scikit-learn.org/stable/modules/clustering.html#k-means
multi_cluster.inertia_

# %%
multi_cluster.cluster_centers_

# %%
multi_inertia_scores = []
for iterable in range(1,11):
        multi_kmeans = KMeans(n_clusters=iterable ).fit(standardized_data)
        multi_inertia_scores.append(multi_kmeans.inertia_)

# %%
plt.pyplot.plot(range(1,11),multi_inertia_scores)
plt.pyplot.scatter(range(1,11),multi_inertia_scores, edgecolors="black")
plt.pyplot.plot(4, multi_inertia_scores[4], marker="x", color="green", linestyle="dashed", linewidth=2, markersize=12)
plt.pyplot.xlabel("Number of Clusters", size=13)
plt.pyplot.ylabel("Inertia Value", size=13)
plt.pyplot.title("Different Inertia Values for Different Number of Clusters", size=17);


# cluster_centriod = pd.DataFrame(multi_cluster.cluster_centers_)
# cluster_centriod.columns = ["w","x","y","z"]
# plt.pyplot.figure(figsize=(10,8))
# plt.pyplot.scatter(x=cluster_centriod["w"], y=cluster_centriod["x"], s=300, color="red", marker="^")
# sns.scatterplot(data=multi_cluster_data, x=cluster_centriod["w"], y=cluster_centriod["x"], palette="sunshine");

# %% [markdown]
# <span style="color:rgb(0,255,127);">Mulitvariate Analysis</span>

# %%
from statsmodels.multivariate.manova import MANOVA
MANOVA_data = encode_cat.set_axis(["Customer_Id", "Age", "Annual_Income", "Spending_Score", "Income_Cluster", "Income_Spending_Cluster", "Gender_Male"], axis=1, inplace=False)
fit_data = MANOVA.from_formula("Annual_Income + Spending_Score ~ Age + Gender_Male", data=MANOVA_data)
print(fit_data.mv_test())

# %% [markdown]
# <span style="transform: rotate(180deg); color:rgb(255,69,0); font-style: oblique; letter-spacing: 1px;">Multivariate 3D Visualization</span>

# %%
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
x = encode_cat[["Annual Income (k$)", "Spending Score (1-100)"]]
y = encode_cat[["Age"]]
x = sm.add_constant(x)
ols_data = sm.OLS(y,x).fit()
ols_data.summary()

# %%
x = encode_cat[["Annual Income (k$)", "Spending Score (1-100)"]]
y = encode_cat[["Gender_Male"]]
x = sm.add_constant(x)
ols_data = sm.OLS(y,x).fit()
ols_data.summary()

# %%
x = encode_cat[["Age","Gender_Male"]]
y = encode_cat[["Annual Income (k$)"]]
x = sm.add_constant(x)
ols_data = sm.OLS(y,x).fit()
ols_data.summary()

# %%
x = encode_cat[["Age","Gender_Male"]]
y = encode_cat[["Spending Score (1-100)"]]
x = sm.add_constant(x)
ols_data = sm.OLS(y,x).fit()
ols_data.summary()

# %%

x = encode_cat[["Age","Gender_Male"]]
y = encode_cat[["Annual Income (k$)"]]
x = sm.add_constant(x)
ols_data = sm.OLS(y, x).fit()

# create the 3d plot 
# Age/Gender_Male grid for 3d plot
xv, yv = np.meshgrid(np.linspace(x.Age.min(), x.Age.max(), num=100), np.linspace(x.Gender_Male.min(), x.Gender_Male.max(), num=100))

# plot the hyperplane by evaluating the parameters on the grid
z = ols_data.params[0] + ols_data.params[1] * xv + ols_data.params[2] * yv

# create matplotlib 3d axes
fig = plt.pyplot.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
# ax = Axes3D(fig, azim=-115, elev=15)

# plot hyperplane
surface = ax.plot_surface(xv, yv, z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

# plot data points - points over the HP are white, points below are black
residual = y["Annual Income (k$)"] - ols_data.predict(x)
ax.scatter(x[residual >= 0].Age, x[residual >= 0].Gender_Male, y[residual >= 0], color="black", alpha=1.0, facecolor="white")
ax.scatter(x[residual < 0].Age, x[residual < 0].Gender_Male, y[residual < 0], color="black", alpha=1.0)

# set axis labels
ax.set_xlabel("Age")
ax.set_ylabel("Gender Male")
ax.set_zlabel("Annual Income (k$)")
ax.set_title("Age and Gender on Annual Income")

# residual = y - ols_data.predict(x)
# ax.scatter(x[residual].Age, x[residual].Gender_Male, y[residual], color="black", alpha=1.0, facecolor="white")
# ax.scatter(x[residual].Age, x[residual].Gender_Male, y[residual], color="black", alpha=1.0)
# ax.scatter(x.Age, x.Gender_Male, y, color="black", alpha=1.0, facecolor="white")
# ax.scatter(x.Age, x.Gender_Male, y, color="black", alpha=1.0)



# %% [markdown]
# ### <span style="background-image:linear-gradient( to left, rgb(40,0,0), rgb(255, 51, 153));"><em>Final Analysis<em></span>

# %% [markdown]
# <span style=color:rgb(102,205,170);>The third and fourth cluster of our bivariate analysis, as illustrated by scatter plot, accounts for the highest annual income, based on average age and gender, are 20, 23, and 31 and is majority female, over 70 percent. Cluster zero of our bivariate analysis with an averge age of 32 years old accounts for the highest annual income and spending score and is mostly female in gender. Individuals with the average age of 23 and gender female accounts for the highest spending score. Individuals 37 years old and older accounted for the highest annual income, but on average were in median range of the spending score, despite having the available income they spent less. This segment of the market could represent an under-engaged gap in the market and marketing stragety. An omnichannel marketing strategy could be developed based the individual psychology and behavorial profile that would motivate that individual to alter their spending pattern. It is worth noting that a high spending score could be attributed to the individual purchasing highly priced items or spending/purchasing patterns based a host of factors such as the availability of wealth and require further marketing research to determine those factors. The geographic area is fixed to our local mall, and we are only targeting demographics based on age and gender, so further marketing research is necessary to determine the psychology and behaviorial patterns of our target markets. The current analysis does provide data about what specific target markets to focus on or not focus on when allocating resources in marketing and further marketing research.</span>


