import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hydralit_components as hc
import plotly.graph_objects as go
import pickle
from streamlit_lottie import st_lottie
import json


# Streamlit Style Settings
def webapp_style():
    hide_streamlit_style = """
                <style>
                    #MainMenu {
                                visibility: none;
                            }
                    footer {
                            visibility: hidden;
                            }
                    footer:after {
                                content:'Made by Zainab Hodroj '; 
                                visibility: visible;
                                display: block;
                                position: relative;
                                text-align: center;
                                padding: 15px;
                                top: 2px;
                                }
    
                </style>
                """
    markdown = st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    return markdown

#defining lottie function to visualize animated pictures
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def upload():

    # Dispaly Upload File Widget
    uploaded = st.file_uploader(label="Upload your own data", type=["csv"])

    # Save the file in internal memory of streamlit
    if 'file' not in st.session_state:
        st.session_state['file'] = None


    st.session_state['file'] = uploaded

    if 'table' not in st.session_state:
        st.session_state['table'] = None 
    
        
    if uploaded is not None:
        st.session_state['table'] = pd.read_csv(uploaded)
        return st.session_state['table']
    else:
        st.session_state['table'] = pd.read_csv('clean_data.csv')
        return st.session_state['table']



#setting configuration of the page and expanding it
st.set_page_config(layout='wide', initial_sidebar_state='collapsed', page_title='Amount Spent Prediction', page_icon="✅")
st.expander('Expander')


#creating menu data which will be used in navigation bar specifying the pages of the dashboard
menu_data = [
    {'label': "Home", 'icon': 'bi bi-house-fill'},
    {'label': 'Data', 'icon': 'bi bi-bar-chart'},
    {'label':"EDA", 'icon':'bi bi-search'},
    {'label':"Amount Spent Prediction", 'icon': 'bi bi-person-fill'},
    ]

over_theme = {'txc_inactive': 'white','menu_background':'rgb(112, 230, 220)', 'option_active':'white'}


#inserting hydralit component: nagivation bar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=False,
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

#editing first page of the dashboard with images, titles, and text
if menu_id == 'Home':
    col1, col2 = st.columns(2)
    #col1.image('churn.png')
    with col1:
        st.title('Amount Spent Prediction')
        st.write('Many companies would benefit immensely by investigating the amount spent on campaigns and taking steps to improve it. Lets look at how we can develop this intelligence using machine learning')
        st.write("Once businesses determine this value, they can determine improve their budget and set up their campaigns accordingly")    
        st.write("You will be able to have hands on experience with a model that can accurately predict the value of the amount spent on campaigns")
        m = st.markdown("""
        <style>
            div.stButton > button:first-child {
            color: #fff;
            background-color: rgb(112, 230, 220);
            }
        </style>""", unsafe_allow_html=True)
        st.write("---")

    with col2:
        lottie_home= load_lottiefile("home.json")
        st_lottie(lottie_home)
        



#BREAK
#editing second page which is about the data
if menu_id == 'Data':
    #add option to choose own data or given data defined earlier
    upload()
    col1, col2 = st.columns([1,1])

    with col1:
        lottie_data= load_lottiefile("exploratory.json")
        st_lottie(lottie_data)

    with col2:
        st.title("Let's Take a Look at the Data")
        st.markdown("""
        <style>
        .change-font {
        font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="change-font">The data includes different objectives and KPIs like reach, impressions, clicks, and other features </p>', unsafe_allow_html=True)
        st.markdown('<p class="change-font">Our goal is to find the variables related to the amount spent and create an ML tool that will help in accuratly predicting the amount spent. </p>', unsafe_allow_html=True)
        data= pd.read_csv(r'clean_data.csv')
        if st.checkbox('Dataset'):
            st.dataframe(data.head(5))
        if st.checkbox('Statistics'):
            st.dataframe(data.describe())
    


df= pd.read_csv(r'clean_data.csv')
#BREAK


# 3rd page Exploratory Data Analysis
if menu_id == 'EDA':
    #KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        #info card of the Number of customers
        Objective=  len(pd.unique(df['Objective']))
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-search'}
        hc.info_card(title='Objectives', content=Objective, theme_override=theme_override)
    with col2:
        #info card of the average transaction value
        #info card of the Number of customers
        Country=  len(pd.unique(df['Country']))
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-map'}
        hc.info_card(title='Country', content=Country, theme_override=theme_override)
    with col3:
        #info card of the average transaction value
        #info card of the Number of customers
        Average_reach =int(df["Reach"].mean())
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-people'}
        hc.info_card(title='Average Reach', content=Average_reach, theme_override=theme_override)

    with col4:
        #info card of the average transaction value
        #info card of the Number of customers
        Average_Impressions =int(df["Impressions"].mean())
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-people-fill'}
        hc.info_card(title='Average_Impressions', content=Average_Impressions, theme_override=theme_override)
            



    #split to plot 2 side-by-side graphs
    col1, col2= st.columns(2)
    #df = df[df['Objective'].isin(gender_selections)]
    #df = df[df['Country'].isin(region_selections)]
    color_discrete_sequence= ['rgb(220,176,242)', ' rgb(254,136,177)',
             'rgb(136,204,238)']
    with col1:
        #Count by Objective 
        data= pd.read_csv('clean_data.csv')
        fig10 = px.bar(data, x=data['Country']
                     )
        fig10.update_layout(title='Count by Country', xaxis_title='Country', yaxis_title='Count')
        st.plotly_chart(fig10)
        
        
        #scatter plot of reach vs. Amount Spent
        fig20 = go.Figure(px.pie(data, values='Amount Spent', names='Objective', title='Amount Spent by Objective',
             color_discrete_sequence= color_discrete_sequence
            ))
        st.plotly_chart(fig20)     
        
         
        
    with col2:
        fig11 = px.bar(data, x=data['Objective']
                     )
        fig11.update_layout(title='Count by Objective', xaxis_title='Objective', yaxis_title='Count')
        st.plotly_chart(fig11) 
                
        
        
        fig5 = go.Figure(px.pie(data, values='Amount Spent', names='Country', title='Amount Spent by Country',
             color_discrete_sequence= color_discrete_sequence
            ))
        st.plotly_chart(fig5)

#BREAK
        
        
if menu_id == 'Amount Spent Prediction':
    # load assets (lotties animation)
    lottie1 = load_lottiefile("predict.json")
    #1 Create header and title
    with st.container():
        st.title("Amount Spent Predictionn")
        
    col1, col2 = st.columns((1,1))
    with col1:
        st.write("We have created a model that targets predicting amount spent on marketing campaign")
        #adding animation and figures to the right column

    with col2:
        st_lottie(lottie1, height=300, width=800, key="mental health")
        #3 create the second part with the input questions
        #add header for the input part
        
    st.write("---")
    st.subheader("Please fill in the following questions to predict the amount spent")

        #adding input options 
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("0:CPA, 1:CPC, 2:CPE, 3:CPL, 4:CPLPV, 5:CPMR,6:CPV ,7:Remarketing")   
        Objective = st.selectbox('What is the objective of your campaign?', 
                                           ('0', '1', '2', '3', '4', '5', '6'))
        st.write("0:Egypt, 1:KSA, 2:Qatar, 3:UAE, 4:na")
        country = st.selectbox('Country', ('0', '1', '2', '3', '4'))
        Reach = st.number_input('Reach',min_value=3, max_value=1316902, step=1)
        Impressions = st.number_input('Impressions',min_value=3, max_value=1438509, step=1)
        Link_clicks = st.number_input('Link clicks',min_value=0, max_value=5997, step=1)
        Landing_page_views = st.number_input('Landing page views',min_value=0, max_value=1198, step=1)
        Post_engagement = st.number_input('Post engagement',min_value=0, max_value=160855, step=1)
    with col2:
        three_second_video_plays = st.number_input('3-second video plays',min_value=0, max_value=154402, step=1)
        ThruPlays = st.number_input('ThruPlays',min_value=0, max_value=70739, step=1)
        Post_comments = st.number_input('Post comments',min_value=0, max_value=370, step=1)
        Post_saves = st.number_input('Post saves',min_value=0, max_value=122, step=1)
        Post_reactions = st.number_input('Post reactions',min_value=0, max_value=6763, step=1)
        Post_shares = st.number_input('Post shares',min_value=0, max_value=483, step=1)
        Video_plays = st.number_input('Video plays at 50%',min_value=0, max_value=75918, step=1)
        Leads = st.number_input('Leads',min_value=0, max_value=171, step=1)
        #4 load model and create new dataframe
        with st.container():
            import pickle
            loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
        # 5 assigning values to the new df
        with st.container():
            new_df = pd.DataFrame({'Objective': [Objective], 'Country': [country], 'Reach': [Reach], 'Impressions': [Impressions], 'Link_clicks': [Link_clicks], 'Landing_page_views': [Landing_page_views], 'Post_engagement': [Post_engagement], 'three_second_video_plays': [three_second_video_plays], 'ThruPlays': [ThruPlays], 'Post_comments': [Post_comments], 'Post_saves': [Post_saves], 'Post_reactions': [Post_reactions], 'Post_shares': [Post_shares], 'Video_plays': [Video_plays], 'Leads': [Leads]})

        # 6 predict_proba(new_df) and create botton for diagnosis
        with col1: 
            if st.button('Predict'):
                predicted = loaded_model.predict(new_df)
                predicted = round(predicted[0], 2)
                st.write('Estimated Amount Spent on the Campaign is {} $.'.format(predicted))

             
webapp_style()       


# Thank you for using our app!
