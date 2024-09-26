import random

import gensim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import pyLDAvis.gensim_models
import regex
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from wordcloud import WordCloud

# 기본 설정 값 정의
DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM = 0.001
DEFAULT_NUM_TOPICS = 6

# NLTK의 불용어 데이터 다운로드
nltk.download("stopwords")

# 데이터셋 정보 사전 정의
# DATASETS = {
#     'Five Years of Elon Musk Tweets': {
#         'path': './data/elonmusk.csv.zip',
#         'column': 'tweet',
#         'url': 'https://www.kaggle.com/vidyapb/elon-musk-tweets-2015-to-2020',
#         'description': (
#             'twint 라이브러리를 사용하여 최근 5년간 Elon Musk의 트윗을 스크랩했습니다. '
#             '이 데이터셋을 통해 공인 인사가 소셜 미디어 플랫폼에서 일반인에게 어떤 영향을 미치는지 '
#             '살펴보고자 했습니다. Tesla가 주로 어떤 주제에 대해 트윗하는지, '
#             'Elon Musk의 트윗이 Tesla 주식에 어떻게 영향을 미치는지 등의 인사이트를 제공하는 노트북을 보고 싶습니다.'
#         )
#     },
#     'Airline Tweets': {
#         'path': './data/Tweets.csv.zip',
#         'column': 'text',
#         'url': 'https://www.kaggle.com/crowdflower/twitter-airline-sentiment',
#         'description': (
#             '각 주요 미국 항공사의 문제점에 대한 감성 분석 작업입니다. '
#             '2015년 2월에 스크랩된 트위터 데이터를 기반으로 기여자들은 먼저 긍정적, 부정적, 중립적인 트윗을 분류한 후, '
#             '"늦은 비행" 또는 "무례한 서비스"와 같은 부정적 이유를 분류하도록 요청받았습니다.'
#         )
#     }
# }

# LDA 모델 옵션을 설정하는 함수
def lda_options():
    return {
        'num_topics': st.number_input(
            '토픽 수',
            min_value=1,
            value=9,
            help='학습 코퍼스에서 추출할 잠재 토픽의 수를 설정합니다.'
        ),
        'chunksize': st.number_input(
            '청크 크기',
            min_value=1,
            value=2000,
            help='각 학습 청크에 사용할 문서의 수를 설정합니다.'
        ),
        'passes': st.number_input(
            '패스 수',
            min_value=1,
            value=1,
            help='학습 중 코퍼스를 통과하는 패스 수를 설정합니다.'
        ),
        'update_every': st.number_input(
            '업데이트 주기',
            min_value=1,
            value=1,
            help='각 업데이트마다 반복할 문서의 수를 설정합니다. 배치 학습을 원할 경우 0으로 설정하고, 반복 학습을 원할 경우 1 이상으로 설정합니다.'
        ),
        'alpha': st.selectbox(
            '𝛼',
            ('symmetric', 'asymmetric', 'auto'),
            help='문서-토픽 분포에 대한 사전값(priori belief)을 설정합니다.'
        ),
        'eta': st.selectbox(
            '𝜂',
            (None, 'symmetric', 'auto'),
            help='토픽-단어 분포에 대한 사전값(priori belief)을 설정합니다.'
        ),
        'decay': st.number_input(
            '𝜅',
            min_value=0.5,
            max_value=1.0,
            value=0.5,
            help='새로운 문서를 검사할 때 이전 람다 값을 얼마나 잊어버릴지 결정하는 (0.5, 1] 사이의 값을 설정합니다.'
        ),
        'offset': st.number_input(
            '𝜏_0',
            value=1.0,
            help='처음 몇 번의 반복에서 학습 속도를 얼마나 늦출지를 제어하는 하이퍼파라미터입니다.'
        ),
        'eval_every': st.number_input(
            '평가 주기',
            min_value=1,
            value=10,
            help='얼마나 자주 퍼플렉서티를 로그로 평가할지를 설정합니다.'
        ),
        'iterations': st.number_input(
            '반복 횟수',
            min_value=1,
            value=50,
            help='코퍼스의 토픽 분포를 추론할 때 최대 반복 횟수를 설정합니다.'
        ),
        'gamma_threshold': st.number_input(
            '𝛾',
            min_value=0.0,
            value=0.001,
            help='감마 파라미터 값의 최소 변화량을 설정하여 반복을 계속할지를 결정합니다.'
        ),
        'minimum_probability': st.number_input(
            '최소 확률',
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            help='이 임계값보다 낮은 확률을 가진 토픽은 필터링됩니다.'
        ),
        'minimum_phi_value': st.number_input(
            '𝜑',
            min_value=0.0,
            value=0.01,
            help='per_word_topics가 True일 경우, 용어 확률의 하한을 설정합니다.'
        ),
        'per_word_topics': st.checkbox(
            '단어별 토픽',
            help='True로 설정하면 모델이 각 단어에 대한 가장 가능성이 높은 토픽 목록과 그 phi 값을 계산합니다.'
        )
    }

# NMF 모델 옵션을 설정하는 함수
def nmf_options():
    return {
        'num_topics': st.number_input(
            '토픽 수',
            min_value=1,
            value=9,
            help='추출할 토픽의 수를 설정합니다.'
        ),
        'chunksize': st.number_input(
            '청크 크기',
            min_value=1,
            value=2000,
            help='각 학습 청크에 사용할 문서의 수를 설정합니다.'
        ),
        'passes': st.number_input(
            '패스 수',
            min_value=1,
            value=1,
            help='학습 코퍼스를 통과하는 전체 패스 수를 설정합니다.'
        ),
        'kappa': st.number_input(
            '𝜅',
            min_value=0.0,
            value=1.0,
            help='경사 하강법의 스텝 크기를 설정합니다.'
        ),
        'minimum_probability': st.number_input(
            '최소 확률',
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            help=(
                'normalize가 True일 경우, 확률이 작은 토픽은 필터링됩니다. '
                'normalize가 False일 경우, 인수가 작은 팩터가 필터링됩니다. '
                'None으로 설정하면 1e-8 값이 사용되어 0을 방지합니다.'
            )
        ),
        'w_max_iter': st.number_input(
            'W 최대 반복',
            min_value=1,
            value=200,
            help='각 배치에서 W를 학습할 최대 반복 횟수를 설정합니다.'
        ),
        'w_stop_condition': st.number_input(
            'W 정지 조건',
            min_value=0.0,
            value=0.0001,
            help='오차 차이가 이 값보다 작아지면 현재 배치의 W 학습을 중지합니다.'
        ),
        'h_max_iter': st.number_input(
            'H 최대 반복',
            min_value=1,
            value=50,
            help='각 배치에서 H를 학습할 최대 반복 횟수를 설정합니다.'
        ),
        'h_stop_condition': st.number_input(
            'H 정지 조건',
            min_value=0.0,
            value=0.001,
            help='오차 차이가 이 값보다 작아지면 현재 배치의 H 학습을 중지합니다.'
        ),
        'eval_every': st.number_input(
            '평가 주기',
            min_value=1,
            value=10,
            help='v - Wh의 l2 노름을 계산할 배치 수를 설정합니다.'
        ),
        'normalize': st.selectbox(
            '정규화',
            (True, False, None),
            help='결과를 정규화할지 여부를 설정합니다.'
        )
    }

# 지원되는 모델 정보 사전 정의
MODELS = {
    'Latent Dirichlet Allocation': {
        'options': lda_options,
        'class': gensim.models.LdaModel,
        'help': 'https://radimrehurek.com/gensim/models/ldamodel.html'
    },
    'Non-Negative Matrix Factorization': {
        'options': nmf_options,
        'class': gensim.models.Nmf,
        'help': 'https://radimrehurek.com/gensim/models/nmf.html'
    }
}

# Matplotlib에서 사용할 색상 목록 생성
COLORS = [color for color in mcolors.XKCD_COLORS.values()]

# 워드클라우드에 사용할 폰트 경로
WORDCLOUD_FONT_PATH = r'./data/Proxima Nova Regular.otf'

# 정규표현식 패턴 정의
EMAIL_REGEX_STR = r'\S*@\S*'
MENTION_REGEX_STR = r'@\S*'
HASHTAG_REGEX_STR = r'#\S+'
URL_REGEX_STR = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'

# 캐싱을 사용하여 선택된 데이터셋의 텍스트 데이터를 불러오는 함수
@st.cache_data()
def generate_texts_df(selected_dataset: str):
    dataset = DATASETS[selected_dataset]
    return pd.read_csv(f'{dataset["path"]}')

# 문서에서 불필요한 요소를 제거하는 전처리 함수
@st.cache_data()
def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    # 이메일, 멘션, 해시태그, URL 제거를 위한 정규표현식 컴파일
    remove_regex = regex.compile(f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')
    texts = [regex.sub(remove_regex, '', text) for text in texts]
    # 간단한 전처리 및 불용어 제거
    docs = [
        [w for w in simple_preprocess(doc, deacc=True) if w not in stopwords.words('english')]
        for doc in texts
    ]
    return docs

# 빅그램(2-그램)을 생성하는 함수
@st.cache_data()
def create_bigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    docs = [bigram_phraser[doc] for doc in docs]
    return docs

# 트라이그램(3-그램)을 생성하는 함수
@st.cache_data()
def create_trigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[docs])
    trigram_phraser = gensim.models.phrases.Phraser(trigram_phrases)
    docs = [trigram_phraser[bigram_phraser[doc]] for doc in docs]
    return docs

# 문서 생성 함수로, 전처리 및 n-그램 생성을 포함
@st.cache_data()
def generate_docs(texts_df: pd.DataFrame, text_column: str, ngrams: str = None):
    docs = denoise_docs(texts_df, text_column)
    if ngrams == 'bigrams':
        docs = create_bigrams(docs)
    if ngrams == 'trigrams':
        docs = create_trigrams(docs)
    return docs

# 워드클라우드를 생성하는 함수
@st.cache_data()
def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud(
        font_path=WORDCLOUD_FONT_PATH,
        width=700,
        height=600,
        background_color='white',
        collocations=collocations
    ).generate(wordcloud_text)
    return wordcloud

# 훈련 데이터를 준비하는 함수
@st.cache_data()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus

# 모델을 훈련하는 함수
@st.cache_data()
def train_model(docs, base_model, **kwargs):
    id2word, corpus = prepare_training_data(docs)
    model = base_model(corpus=corpus, id2word=id2word, **kwargs)
    return id2word, corpus, model

# 세션 상태를 초기화하는 함수
def clear_session_state():
    for key in (
        'model_kwargs',
        'id2word',
        'corpus',
        'model',
        'previous_perplexity',
        'previous_coherence_model_value'
    ):
        if key in st.session_state:
            del st.session_state[key]

# 퍼플렉서티를 계산하는 함수
def calculate_perplexity(model, corpus):
    return np.exp2(-model.log_perplexity(corpus))

# 코히런스 점수를 계산하는 함수
def calculate_coherence(model, corpus, coherence):
    coherence_model = CoherenceModel(model=model, corpus=corpus, coherence=coherence)
    return coherence_model.get_coherence()

# 배경색에 따라 텍스트 색상을 결정하는 함수
@st.cache_data()
def white_or_black_text(background_color):
    # 배경색에 따라 검정 또는 흰색 텍스트 선택
    red = int(background_color[1:3], 16)
    green = int(background_color[3:5], 16)
    blue = int(background_color[5:], 16)
    return 'black' if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 else 'white'

# 퍼플렉서티 섹션을 표시하는 함수
def perplexity_section():
    with st.spinner('퍼플렉서티 계산 중...'):
        perplexity = calculate_perplexity(st.session_state.model, st.session_state.corpus)
    key = 'previous_perplexity'
    delta = f'{perplexity - st.session_state[key]:.4}' if key in st.session_state else None
    st.metric(
        label='퍼플렉서티',
        value=f'{perplexity:.4f}',
        delta=delta,
        delta_color='inverse'
    )
    st.markdown('참고: https://en.wikipedia.org/wiki/Perplexity')
    st.latex(r'Perplexity = \exp\left(-\frac{\sum_d \log(p(w_d|\Phi, \alpha))}{N}\right)')

# 코히런스 섹션을 표시하는 함수
def coherence_section():
    with st.spinner('코히런스 점수 계산 중...'):
        coherence = calculate_coherence(st.session_state.model, st.session_state.corpus, 'u_mass')
    key = 'previous_coherence_model_value'
    delta = f'{coherence - st.session_state[key]:.4f}' if key in st.session_state else None
    st.metric(
        label='코히런스 점수',
        value=f'{coherence:.4f}',
        delta=delta
    )
    st.session_state[key] = coherence
    st.markdown('참고: http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf')
    st.latex(
        r'C_{UMass} = \frac{2}{N \cdot (N - 1)}\sum_{i=2}^N\sum_{j=1}^{i-1}\log\frac{P(w_i, w_j) + \epsilon}{P(w_j)}'
    )

# 저차원 투영을 훈련하는 함수
@st.cache_data()
def train_projection(projection, n_components, df):
    if projection == 'PCA':
        projection_model = PCA(n_components=n_components)
    elif projection == 'T-SNE':
        projection_model = TSNE(n_components=n_components)
    elif projection == 'UMAP':
        projection_model = UMAP(n_components=n_components)
    else:
        raise ValueError(f'알 수 없는 투영 방식: {projection}')
    return projection_model.fit_transform(df)

# 메인 함수 실행
if __name__ == '__main__':
    # 페이지 설정
    st.set_page_config(
        page_title='주제 모델링',
        page_icon='./data/favicon.png',
        layout='wide'
    )

    # 사이드바의 전처리 옵션 폼
    preprocessing_options = st.sidebar.form('preprocessing-options')
    with preprocessing_options:
        st.header('전처리 옵션')
        ngrams = st.selectbox(
            'N-그램',
            [None, 'bigrams', 'trigrams'],
            help='단어의 결합 정도를 설정합니다. 예: 바이그램, 트라이그램'
        )
        st.form_submit_button('전처리 실행')

    # 사이드바의 시각화 옵션 폼
    visualization_options = st.sidebar.form('visualization-options')
    with visualization_options:
        st.header('시각화 옵션')
        collocations = st.checkbox(
            '워드클라우드 콜로케이션 활성화',
            help='워드클라우드에서 구문을 표시할 수 있도록 콜로케이션을 활성화합니다.'
        )
        highlight_probability_minimum = st.select_slider(
            '하이라이트 확률 최소값',
            options=[10 ** exponent for exponent in range(-10, 1)],
            value=DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM,
            help='_토픽 하이라이트 문장_ 시각화에서 단어를 색상으로 하이라이트하기 위한 최소 토픽 확률을 설정합니다.'
        )
        st.form_submit_button('적용')

    # 애플리케이션 제목 및 설명
    st.title('토픽 모델링')
    # st.header('주제 모델링이란?')
    # with st.expander('히어로 이미지'):
    #     st.image('./data/is-this-a-topic-modeling.jpg', caption='아니요 ... 아닙니다 ...', use_column_width=True)
    # st.markdown(
    #     '주제 모델링은 다양한 통계적 학습 방법을 포함하는 포괄적인 용어입니다. '
    #     '이 방법들은 문서를 일련의 주제로, 그 주제를 일련의 단어로 설명합니다. '
    #     '가장 일반적으로 사용되는 두 가지 방법은 잠재 디리클레 할당(LDA)과 비음수 행렬 분해(NMF)입니다. '
    #     '추가적인 한정어 없이 사용될 때, 이 접근 방식은 일반적으로 비지도 학습으로 간주되지만, '
    #     '준지도 및 지도 학습 변형도 존재합니다.'
    # )

    # # 추가 세부 정보 확장
    # with st.expander('추가 세부 정보'):
    #     st.markdown('목표는 행렬 분해로 볼 수 있습니다.')
    #     st.image('./data/mf.png', use_column_width=True)
    #     st.markdown('이 분해는 단어를 기준으로 문서를 직접 특성화하는 것보다 훨씬 효율적으로 만듭니다.')
    #     st.markdown('LDA와 NMF에 대한 자세한 정보는 '
    #                 'https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation 및 '
    #                 'https://en.wikipedia.org/wiki/Non-negative_matrix_factorization 에서 확인할 수 있습니다.')

    # 데이터셋 섹션
    st.header('데이터셋')
    # # st.markdown('설명을 위해 미리 로드된 몇 가지 작은 예제 데이터셋을 사용합니다.')
    # selected_dataset = st.selectbox(
    #     '데이터셋',
    #     [None, *sorted(list(DATASETS.keys()))],
    #     on_change=clear_session_state
    # )
    # if not selected_dataset:
    #     st.write('데이터셋을 선택하여 계속하세요 ...')
    #     st.stop()

    # # 데이터셋 설명 확장
    # with st.expander('데이터셋 설명'):
    #     st.markdown(DATASETS[selected_dataset]['description'])
    #     st.markdown(DATASETS[selected_dataset]['url'])

    # 파일 업로더 추가
    st.header('파일 업로드')
    uploaded_files = st.file_uploader(
        "docx, pdf, txt 파일을 업로드하세요",
        type=['docx', 'pdf', 'txt'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.warning('처리할 파일을 업로드해주세요.')
        st.stop()

    # 업로드된 파일에서 텍스트를 추출하는 함수
    def extract_text(file):
        if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # docx 파일 처리
            doc = Document(file)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif file.type == "application/pdf":
            # pdf 파일 처리
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text
        elif file.type == "text/plain":
            # txt 파일 처리
            return file.getvalue().decode("utf-8")
        else:
            return ""
        
    # 업로드된 파일에서 텍스트 추출 및 데이터프레임 생성
    texts = []
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        if text:
            texts.append({
                '파일명': uploaded_file.name,
                '텍스트': text
            })

    if not texts:
        st.error('업로드된 파일에서 텍스트를 추출할 수 없습니다.')
        st.stop()

    texts_df = pd.DataFrame(texts)


    # 업로드된 데이터의 텍스트 컬럼 지정
    text_column = '텍스트'
    docs = generate_docs(texts_df, text_column, ngrams=ngrams)

    # 샘플 문서 확장
    with st.expander('샘플 문서'):
        sample_texts = texts_df[text_column].sample(5).values.tolist()
        for index, text in enumerate(sample_texts):
            st.markdown(f'**{index + 1}**: _{text}_')

    # 워드클라우드 시각화 확장
    with st.expander('빈도 기반 코퍼스 워드클라우드'):
        wc = generate_wordcloud(docs)
        st.image(wc.to_image(), caption='데이터셋 워드클라우드 (토픽 모델링 아님)', use_column_width=True)
        st.markdown('문서 전처리 후 남은 단어들입니다.')

    # 문서 단어 수 분포 시각화 확장
    with st.expander('문서 단어 수 분포'):
        len_docs = [len(doc) for doc in docs]
        df_len_docs = pd.DataFrame(len_docs, columns=['문서당 단어 수'])
        
        # Plotly Express를 사용한 인터랙티브 히스토그램 생성
        fig = px.histogram(
            df_len_docs,
            x='문서당 단어 수',
            nbins=50,  # 원하는 빈의 수로 조정 가능
            labels={'문서당 단어 수': '문서당 단어 수', 'count': '빈도'},
            title='문서 단어 수 분포',
            hover_data={'문서당 단어 수': True, 'count': True},
            opacity=0.75,
            color_discrete_sequence=['#636EFA']
        )
        
        # 레이아웃 업데이트 (옵션)
        fig.update_layout(
            xaxis_title='Words in document',
            yaxis_title='Count',
            bargap=0.1,  # 막대 사이 간격
            template='plotly_white'  # 흰색 배경 템플릿
        )
        
        # Plotly 히스토그램을 Streamlit에 표시
        st.plotly_chart(fig, use_container_width=True)


    # 모델 선택 섹션
    model_key = st.sidebar.selectbox(
        '모델',
        [None, *list(MODELS.keys())],
        on_change=clear_session_state
    )
    model_options = st.sidebar.form('model-options')
    if not model_key:
        with st.sidebar:
            st.write('모델을 선택하여 계속하세요 ...')
        st.stop()
    with model_options:
        st.header('모델 옵션')
        model_kwargs = MODELS[model_key]['options']()
        st.session_state['model_kwargs'] = model_kwargs
        train_model_clicked = st.form_submit_button('모델 학습')

    # 모델 학습 버튼 클릭 시
    if train_model_clicked:
        with st.spinner('모델 학습 중...'):
            id2word, corpus, model = train_model(
                docs,
                MODELS[model_key]['class'],
                **st.session_state.model_kwargs
            )
        st.session_state.id2word = id2word
        st.session_state.corpus = corpus
        st.session_state.model = model

    # 모델이 세션 상태에 없는 경우 중지
    if 'model' not in st.session_state:
        st.stop()

    # 모델 정보 섹션
    st.header('모델')
    st.write(type(st.session_state.model).__name__)
    st.write(st.session_state.model_kwargs)

    # 모델 결과 섹션
    st.header('모델 결과')

    # 토픽 추출
    topics = st.session_state.model.show_topics(
        formatted=False,
        num_words=50,
        num_topics=st.session_state.model_kwargs['num_topics'],
        log=False
    )
    with st.expander('토픽 단어 가중치 요약'):
        topic_summaries = {}
        for topic in topics:
            topic_index = topic[0]
            topic_word_weights = topic[1]
            # 상위 10개의 단어와 가중치를 요약
            topic_summaries[topic_index] = ' + '.join(
                f'{weight:.3f} * {word}' for word, weight in topic_word_weights[:10]
            )
        for topic_index, topic_summary in topic_summaries.items():
            st.markdown(f'**토픽 {topic_index}**: _{topic_summary}_')

    # 랜덤 색상 샘플링
    colors = random.sample(COLORS, k=model_kwargs['num_topics'])
    with st.expander('상위 N 토픽 키워드 워드클라우드'):
        cols = st.columns(3)
        for index, topic in enumerate(topics):
            wc = WordCloud(
                font_path=WORDCLOUD_FONT_PATH,
                width=700,
                height=600,
                background_color='white',
                collocations=collocations,
                prefer_horizontal=1.0,
                color_func=lambda *args, **kwargs: colors[index]
            )
            with cols[index % 3]:
                wc.generate_from_frequencies(dict(topic[1]))
                st.image(wc.to_image(), caption=f'토픽 #{index}', use_column_width=True)

    # 토픽 하이라이트 문장 시각화 섹션
    with st.expander('토픽 하이라이트 문장'):
        sample = texts_df.sample(10)
        for index, row in sample.iterrows():
            html_elements = []
            for token in row[text_column].split():
                if st.session_state.id2word.token2id.get(token) is None:
                    # 단어가 사전에 없으면 취소선 적용
                    html_elements.append(f'<span style="text-decoration:line-through;">{token}</span>')
                else:
                    term_topics = st.session_state.model.get_term_topics(token, minimum_probability=0)
                    topic_probabilities = [term_topic[1] for term_topic in term_topics]
                    max_topic_probability = max(topic_probabilities) if topic_probabilities else 0
                    if max_topic_probability < highlight_probability_minimum:
                        # 확률이 최소값보다 낮으면 일반 텍스트
                        html_elements.append(token)
                    else:
                        # 최대 확률을 가진 토픽의 색상으로 하이라이트
                        max_topic_index = topic_probabilities.index(max_topic_probability)
                        max_topic = term_topics[max_topic_index]
                        background_color = colors[max_topic[0]]
                        color = white_or_black_text(background_color)
                        html_elements.append(
                            f'<span style="background-color: {background_color}; color: {color}; opacity: 0.5;">{token}</span>'
                        )
            st.markdown(f'문서 #{index}: {" ".join(html_elements)}', unsafe_allow_html=True)

    # 모델 메트릭 섹션
    has_log_perplexity = hasattr(st.session_state.model, 'log_perplexity')
    with st.expander('메트릭'):
        if has_log_perplexity:
            left_column, right_column = st.columns(2)
            with left_column:
                perplexity_section()
            with right_column:
                coherence_section()
        else:
            coherence_section()

    # 저차원 투영 시각화 섹션
    with st.expander('저차원 투영'):
        with st.form('projections-form'):
            left_column, right_column = st.columns(2)
            projection = left_column.selectbox(
                '투영 방식',
                ['PCA', 'T-SNE', 'UMAP'],
                help='데이터를 저차원 공간에 투영하는 방식을 선택합니다.'
            )
            plot_type = right_column.selectbox(
                '플롯 유형',
                ['2D', '3D'],
                help='2차원 또는 3차원 플롯을 선택합니다.'
            )
            n_components = 3  # 기본적으로 3차원 투영
            columns = [f'proj{i}' for i in range(1, 4)]
            generate_projection_clicked = st.form_submit_button('투영 생성')

        if generate_projection_clicked:
            topic_weights = []
            for index, topic_weight in enumerate(st.session_state.model[st.session_state.corpus]):
                weight_vector = [0] * int(st.session_state.model_kwargs['num_topics'])
                for topic, weight in topic_weight:
                    weight_vector[topic] = weight
                topic_weights.append(weight_vector)
            df = pd.DataFrame(topic_weights)
            dominant_topic = df.idxmax(axis='columns').astype('string')
            dominant_topic_percentage = df.max(axis='columns')
            df = df.assign(
                dominant_topic=dominant_topic,
                dominant_topic_percentage=dominant_topic_percentage,
                text=texts_df[text_column]
            )
            with st.spinner('투영 학습 중...'):
                projections = train_projection(
                    projection,
                    n_components,
                    df.drop(columns=['dominant_topic', 'dominant_topic_percentage', 'text']).add_prefix('topic_')
                )
            data = pd.concat([df, pd.DataFrame(projections, columns=columns)], axis=1)

            # Plotly 옵션 설정
            px_options = {
                'color': 'dominant_topic',
                'size': 'dominant_topic_percentage',
                'hover_data': ['dominant_topic', 'dominant_topic_percentage', 'text']
            }
            if plot_type == '2D':
                fig = px.scatter(data, x='proj1', y='proj2', **px_options)
                st.plotly_chart(fig)
                fig = px.scatter(data, x='proj1', y='proj3', **px_options)
                st.plotly_chart(fig)
                fig = px.scatter(data, x='proj2', y='proj3', **px_options)
                st.plotly_chart(fig)
            elif plot_type == '3D':
                fig = px.scatter_3d(data, x='proj1', y='proj2', z='proj3', **px_options)
                st.plotly_chart(fig)

    # pyLDAvis 시각화 버튼 및 섹션
    if hasattr(st.session_state.model, 'inference'):  # gensim Nmf는 'inference' 속성이 없어 pyLDAvis 실패
        if st.button('pyLDAvis 생성'):
            with st.spinner('pyLDAvis 시각화 생성 중...'):
                py_lda_vis_data = pyLDAvis.gensim_models.prepare(
                    st.session_state.model,
                    st.session_state.corpus,
                    st.session_state.id2word
                )
                py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)
            with st.expander('pyLDAvis', expanded=True):
                st.markdown(
                    'pyLDAvis는 학습된 주제 모델의 주제를 해석하는 데 도움이 되는 대화형 웹 기반 시각화를 제공합니다. '
                    '이 패키지는 학습된 LDA 주제 모델에서 정보를 추출하여 시각화를 생성합니다.'
                )
                st.markdown('https://github.com/bmabey/pyLDAvis')
                components.html(py_lda_vis_html, width=1300, height=800)
