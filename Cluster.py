import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn.pipeline import Pipeline

with open("D:/DS ML&AI/Capstone4/env/Scripts/rfm_kmeans_pipeline.pkl", "rb") as f:
                loaded_pipeline = pickle.load(f)

# with open("D:/DS ML&AI/Capstone4/env/Scripts/item_similarity.pkl", "rb") as f:
#     item_similarity_df = pickle.load(f)

item_similarity_df=pd.read_csv("D:/DS ML&AI/Capstone4/env/Scripts/item_similarity_matrix.csv",index_col=0)

item_similarity_df.index = item_similarity_df.index.str.strip()
item_similarity_df.columns = item_similarity_df.columns.str.strip()

    
with st.sidebar:
     with st.container(border=True):
         st.header('Navigation')
         options = ["Clustring","Recommendation"]
         selection = st.pills('Options', options,default='Clustring',selection_mode="single",width='stretch')

if selection=="Clustring":
    st.title('Customer Segmentation')
    st.divider()
    recency=st.number_input('Recency(days since last purchase)',min_value=0,max_value=500,step=1)
    frequency=st.number_input('Frequency(Number of purchases)',min_value=0,max_value=500,step=1)
    monetry=st.number_input('Monetry(Total Spend)')
    st.divider()
    butt=st.button('Predict Customer Segment')
    if butt:
        new_customer = pd.DataFrame([{"Recency":recency,"Frequency":frequency,"Monetary":monetry}])
        predicted_cluster = loaded_pipeline.predict(new_customer)
        if predicted_cluster==0:
            st.success(f"Customer belongs to _**Regular**_ Customer Group")
        elif predicted_cluster==1:
            st.success(f"Customer belongs to _**At Risk**_ Customer Group")
        elif predicted_cluster==2:
            st.success(f"Customer belongs to _**Supreme Value**_ Customer Group")        
        elif predicted_cluster==3:
            st.success(f"Customer belongs to _**Occasional**_ Customer Group")
        elif predicted_cluster==4:
            st.success(f"Customer belongs to _**High Value**_ Customer Group")   


if selection=="Recommendation":
    st.title('Product Recommender')
    st.divider()
    product=st.text_input("Enter Product Name",max_chars=150, help="eg:CREAM CUPID HEARTS COAT HANGER")
    button=st.button('Get Recommendations')
    if button:
        @st.cache_data
        def recommend_similar_items(product_name, top_n=5):
            if product_name not in item_similarity_df.index:
                return f"Product '{product_name}' not found in dataset."
            similar_items = item_similarity_df.loc[product_name].sort_values(ascending=False)[1:top_n+1]
            similar_items_df = similar_items.reset_index()
            similar_items_df.columns = ["Recommended Products","score"]
            return similar_items_df["Recommended Products"]
            #return list(similar_items)

        recom=recommend_similar_items(product,top_n=5)
        #st.table(recom)
        st.dataframe(recom,hide_index=True)
        #st.write(recom)
        # df1 = pd.DataFrame({"Recommended Products": [recom]})
        # st.dataframe(df1)
        
        

    
    
        