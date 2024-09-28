import os
import random
import regex
import numpy as np
import pandas as pd
import streamlit as st
import gensim
import matplotlib.colors as mcolors
import nltk
import plotly.express as px
import pyLDAvis.gensim_models
from docx import Document  # docx 파일 처리를 위한 라이브러리
import PyPDF2  # pdf 파일 처리를 위한 라이브러리
from gensim import corpora
from gensim.models import CoherenceModel, Phrases, LdaModel, Nmf
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from io import BytesIO
import re  # 특수 문자 제거를 위한 라이브러리

# 복합어 사전 정의
REPLACEMENTS = {
    "cyber security": "cybersecurity",
}

# 기본 설정 값 정의
DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM = 0.001
DEFAULT_NUM_TOPICS = 6

# NLTK의 불용어 및 WordNet 데이터 다운로드
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")  # WordNet Lemmatizer를 위한 리소스

# 정규표현식 패턴 정의
EMAIL_REGEX_STR = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
MENTION_REGEX_STR = r'@\w+'
HASHTAG_REGEX_STR = r'#\w+'
URL_REGEX_STR = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# 워드클라우드에 사용할 폰트 경로
WORDCLOUD_FONT_PATH = os.path.join('.', 'data', 'Proxima Nova Regular.otf')

# 지원되는 모델 정보 사전 정의
def lda_options():
    with st.sidebar.form('lda-options'):
        st.header('LDA 옵션')
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
            ),
            'submit': st.form_submit_button('적용')
        }

def nmf_options():
    with st.sidebar.form('nmf-options'):
        st.header('NMF 옵션')
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
            ),
            'submit': st.form_submit_button('적용')
        }

# 모델 정보 사전 정의
MODELS = {
    'Latent Dirichlet Allocation': {
        'options': lda_options,
        'class': LdaModel,
        'help': 'https://radimrehurek.com/gensim/models/ldamodel.html'
    },
    'Non-Negative Matrix Factorization': {
        'options': nmf_options,
        'class': Nmf,
        'help': 'https://radimrehurek.com/gensim/models/nmf.html'
    }
}

# Matplotlib에서 사용할 색상 목록 생성
COLORS = [color for color in mcolors.XKCD_COLORS.values()]

# 파일 업로드 후 텍스트를 추출하는 함수
def extract_text(file):
    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # docx 파일 처리
        try:
            doc = Document(file)
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"docx 파일 처리 중 오류 발생: {file.name}\n오류 메시지: {e}")
            return ""
    elif file.type == "application/pdf":
        # pdf 파일 처리
        try:
            reader = PyPDF2.PdfReader(file)
            text = ''
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                except Exception as e:
                    st.warning(f"암호화된 PDF 파일을 처리할 수 없습니다: {file.name}")
                    return ""
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + '\n'
            return text
        except Exception as e:
            st.error(f"PDF 파일 처리 중 오류 발생: {file.name}\n오류 메시지: {e}")
            return ""
    elif file.type == "text/plain":
        # txt 파일 처리
        try:
            return file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file.getvalue().decode("latin-1")
            except UnicodeDecodeError as e:
                st.error(f"txt 파일 인코딩을 감지할 수 없습니다: {file.name}\n오류 메시지: {e}")
                return ""
    else:
        st.warning(f"지원되지 않는 파일 형식: {file.name}")
        return ""

# 업로드된 파일에서 데이터프레임 생성
@st.cache_data(show_spinner=False, ttl=600)
def create_texts_df(uploaded_files):
    texts = []
    for uploaded_file in uploaded_files:
        text = extract_text(uploaded_file)
        if text:
            # 단락 단위로 분할
            paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
            for para in paragraphs:
                texts.append({
                    '파일명': uploaded_file.name,
                    '단락': para
                })
    return pd.DataFrame(texts)

# 빅그램(2-그램)을 생성하는 함수
@st.cache_data(show_spinner=False, ttl=600)
def create_bigrams(docs):
    bigram_phrases = Phrases(docs, min_count=5, threshold=100)
    bigram_phraser = Phraser(bigram_phrases)
    docs = [bigram_phraser[doc] for doc in docs]
    return docs

# 트라이그램(3-그램)을 생성하는 함수
@st.cache_data(show_spinner=False, ttl=600)
def create_trigrams(docs):
    bigram_phrases = Phrases(docs, min_count=5, threshold=100)
    bigram_phraser = Phraser(bigram_phrases)
    trigram_phrases = Phrases((bigram_phraser[doc] for doc in docs), min_count=5, threshold=100)
    trigram_phraser = Phraser(trigram_phrases)
    docs = [trigram_phraser[bigram_phraser[doc]] for doc in docs]
    return docs

# 복합어 처리를 위한 n-그램 모델 설정
def build_ngram_model(docs, min_count=5, threshold=100):
    bigram = Phrases(docs, min_count=min_count, threshold=threshold)
    return Phraser(bigram)

lemmatizer = WordNetLemmatizer()

def replace_keywords(docs, replacements=REPLACEMENTS):
    """
    문서 내의 특정 키워드를 표준화된 형태로 대체합니다.  
    Parameters:
        docs (list of list of str): 토큰화된 문서 리스트.
        replacements (dict): 대체할 키워드 사전.  
    Returns:
        list of list of str: 대체된 문서 리스트.
    """
    new_docs = []
    for doc in docs:
        new_doc = []
        skip = False
        i = 0
        while i < len(doc):
            if skip:
                skip = False
                i += 1
                continue
            # 최대 키워드 길이에 맞춰 슬라이싱 (여기서는 최대 2단어)
            replaced = False
            for phrase, replacement in replacements.items():
                phrase_tokens = phrase.split()
                phrase_length = len(phrase_tokens)
                if i + phrase_length <= len(doc) and doc[i:i+phrase_length] == phrase_tokens:
                    new_doc.append(replacement)
                    skip = True
                    replaced = True
                    break
            if not replaced:
                new_doc.append(doc[i])
            i += 1
        new_docs.append(new_doc)
    return new_docs

# 문서에서 불필요한 요소를 제거하고 전처리하는 함수
@st.cache_data(show_spinner=False, ttl=600)
def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    preprocessed_texts = []
    for text in texts:
        # 이메일, 멘션, 해시태그, URL 제거
        remove_regex = regex.compile(f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')
        text = regex.sub(remove_regex, '', text)
        preprocessed_texts.append(text)

    additional_stopwords = {'he', 'she', 'it', 'they', 'we', 'us'}
    all_stopwords = set(stopwords.words('english')).union(additional_stopwords)

    # 불용어 제거 및 표제어 추출
    docs = [
        [lemmatizer.lemmatize(w) for w in simple_preprocess(doc, deacc=True) if w not in all_stopwords]
        for doc in preprocessed_texts
    ]

    # 빅그램 생성
    bigram_model = build_ngram_model(docs)
    docs = [bigram_model[doc] for doc in docs]

    # 특정 빅그램을 단일 단어로 변환 대체 함수 적용
    docs = replace_keywords(docs)

    return docs

# 문서 생성 함수로, 전처리 및 n-그램 생성을 포함
@st.cache_data(show_spinner=False, ttl=600)
def generate_docs(texts_df: pd.DataFrame, text_column: str, ngrams: str = None):
    docs = denoise_docs(texts_df, text_column)
    if ngrams == 'bigrams':
        docs = create_bigrams(docs)
    elif ngrams == 'trigrams':
        docs = create_trigrams(docs)
    return docs

# 워드클라우드를 생성하는 함수
@st.cache_data(show_spinner=False, ttl=600)
def generate_wordcloud(docs, collocations: bool = False, width=1400, height=1200):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud(
        font_path=WORDCLOUD_FONT_PATH,
        width=width,
        height=height,
        background_color='white',
        collocations=collocations
    ).generate(wordcloud_text)
    return wordcloud

# 훈련 데이터를 준비하는 함수
@st.cache_data(show_spinner=False, ttl=600)
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus

# 모델을 훈련하는 함수
def train_model(docs, base_model, **kwargs):
    id2word, corpus = prepare_training_data(docs)
    model = base_model(corpus=corpus, id2word=id2word, **kwargs)
    return id2word, corpus, model

# 퍼플렉서티를 계산하는 함수
def calculate_perplexity(model, corpus):
    return np.exp2(-model.log_perplexity(corpus))

# 코히런스 점수를 계산하는 함수
def calculate_coherence(model, corpus, coherence):
    coherence_model = CoherenceModel(model=model, corpus=corpus, coherence=coherence)
    return coherence_model.get_coherence()

# 퍼플렉서티 섹션을 표시하는 함수
def perplexity_section(model, corpus):
    with st.spinner('퍼플렉서티 계산 중...'):
        perplexity = calculate_perplexity(model, corpus)
    key = 'previous_perplexity'
    if key in st.session_state:
        delta = f'{perplexity - st.session_state[key]:.4}'
    else:
        delta = None
    st.metric(
        label='퍼플렉서티',
        value=f'{perplexity:.4f}',
        delta=delta,
        delta_color='inverse'
    )
    st.markdown('참고: https://en.wikipedia.org/wiki/Perplexity')
    st.latex(r'Perplexity = \exp\left(-\frac{\sum_d \log(p(w_d|\Phi, \alpha))}{N}\right)')
    st.session_state[key] = perplexity

# 코히런스 섹션을 표시하는 함수
def coherence_section(model, corpus):
    with st.spinner('코히런스 점수 계산 중...'):
        coherence = calculate_coherence(model, corpus, 'u_mass')
    key = 'previous_coherence_model_value'
    if key in st.session_state:
        delta = f'{coherence - st.session_state[key]:.4f}'
    else:
        delta = None
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

# 메트릭 섹션을 표시하는 함수
def metrics_section(model, corpus):
    coherence_section(model, corpus)
    if hasattr(model, 'log_perplexity'):
        perplexity_section(model, corpus)

# 배경색에 따라 텍스트 색상을 결정하는 함수
@st.cache_data(show_spinner=False, ttl=600)
def white_or_black_text(background_color):
    # 배경색에 따라 검정 또는 흰색 텍스트 선택
    red = int(background_color[1:3], 16)
    green = int(background_color[3:5], 16)
    blue = int(background_color[5:], 16)
    return 'black' if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 else 'white'

#####################
### 메인 함수 실행 ###
#####################

# Streamlit 앱 실행
def main():
    # 페이지 설정
    st.set_page_config(
        page_title='토픽모델링',
        page_icon='./data/favicon.png',
        layout='wide'
    )

    # 세션 상태 초기화
    if 'model_kwargs' not in st.session_state:
        st.session_state['model_kwargs'] = {}
    if 'ngrams' not in st.session_state:
        st.session_state['ngrams'] = None
    if 'collocations' not in st.session_state:
        st.session_state['collocations'] = False
    if 'highlight_probability_minimum' not in st.session_state:
        st.session_state['highlight_probability_minimum'] = DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM
    if 'colors' not in st.session_state:
        st.session_state['colors'] = []

    # 사이드바의 전처리 옵션 폼
    preprocessing_options = st.sidebar.form('preprocessing-options')
    with preprocessing_options:
        st.header('전처리 옵션')
        st.session_state.ngrams = st.selectbox(
            'N-그램',
            [None, 'bigrams', 'trigrams'],
            help='단어의 결합 정도를 설정합니다. 예: 바이그램, 트라이그램'
        )
        preprocessing_submit = st.form_submit_button('전처리 실행')

    # 사이드바의 시각화 옵션 폼
    visualization_options = st.sidebar.form('visualization-options')
    with visualization_options:
        st.header('시각화 옵션')
        st.session_state.collocations = st.checkbox(
            '워드클라우드 콜로케이션 활성화',
            help='워드클라우드에서 구문을 표시할 수 있도록 콜로케이션을 활성화합니다.'
        )
        st.session_state.highlight_probability_minimum = st.select_slider(
            '하이라이트 확률 최소값',
            options=[10 ** exponent for exponent in range(-10, 1)],
            value=DEFAULT_HIGHLIGHT_PROBABILITY_MINIMUM,
            help='_토픽 하이라이트 문장_ 시각화에서 단어를 색상으로 하이라이트하기 위한 최소 토픽 확률을 설정합니다.'
        )
        visualization_submit = st.form_submit_button('적용')

    # 사이드바의 모델 선택
    model_selection = st.sidebar.selectbox(
        '모델 선택',
        list(MODELS.keys()),
        help='토픽 모델링에 사용할 알고리즘을 선택합니다.'
    )

    # 사이드바의 모델 옵션 폼
    model_options = MODELS[model_selection]['options']()

    # 애플리케이션 제목 및 설명
    st.title('Topic Modeling')
    st.subheader('토픽모델링이란?')
    st.markdown(
        """
문서들의 집합에서 특정한 주제를 찾아내기 위한 자연어 처리 기술(NLP)로,  
특정 주제에 관한 문서에서는 특정 단어가 자주 등장할 것이라는 직관에서 시작된 기술입니다.  
예를 들어서 특정한 문서의 주제가 음식이라면, 음식의 종류, 음식의 재료 등을 나타내는 단어가 다른 문서에 비해 많이 등장한다고 보고,  
그 특정한 주제를 찾아내기 위한 접근법입니다.
"""
    )
    st.markdown(
        """
토픽 모델링은 특히 텍스트 마이닝 기법 중에서도 가장 많이 활용되는 기법 중 하나입니다.  
토픽 모델링은 다시 두 가지의 방법으로 구분되는데, 하나는 잠재 의미 분석(LSA; Latent Semantic Analysis)이고, 다른 하나는 잠재 디리클레 할당(LDA; Latent Dirichlet Allocation)입니다.
"""
    )

    # 추가 세부 정보 확장
    with st.expander('잠재 디리클레 할당'):
        st.image('data/LDA.png')
        st.markdown(
            'LDA(Latent Dirichlet Allocation)는 확률적 그래픽 모델로, 문서 집합 내에 숨겨진 토픽을 발견하기 위해 각 문서를 여러 토픽의 혼합으로 보고, 각 토픽을 단어의 확률 분포로 정의합니다. LDA는 문서 내 단어들이 특정 토픽에서 생성될 확률을 추정하여, 문서와 토픽 간의 관계를 확률적으로 모델링합니다. 이를 통해 문서 집합의 잠재적 구조를 이해하고, 주제별로 문서를 분류하거나 유사한 문서를 찾는 데 유용하게 사용됩니다.'
        )

    with st.expander('비음수 행렬 인수분해 기법'):
        st.image('data/NMF.png')
        st.markdown(
            'NMF(Non-negative Matrix Factorization)는 비음수 행렬 인수분해 기법으로, 단어-문서 행렬을 두 개의 저차원 비음수 행렬로 분해하여 토픽을 추출합니다. NMF는 원래 행렬을 단어-토픽 행렬과 토픽-문서 행렬로 분해함으로써, 각 토픽을 단어들의 가중치 조합으로 해석할 수 있게 합니다. 이 방법은 계산이 비교적 간단하고 빠르며, 비음수 제약으로 인해 결과 해석이 용이하여 텍스트 마이닝에서 널리 사용됩니다. 다만, 초기화에 민감하고 확률적 해석이 부족하다는 단점이 있습니다.'
        )

    # 파일 업로더 추가 (단일 파일 업로드로 변경)
    st.subheader('파일 업로드')
    uploaded_file = st.file_uploader(
        "docx, pdf, txt 파일을 업로드하세요",
        type=['docx', 'pdf', 'txt'],
        accept_multiple_files=False  # 하나의 파일만 업로드 가능하도록 설정
    )

    if not uploaded_file:
        st.warning('처리할 파일을 업로드해주세요.')
        st.stop()

    # 파일 이름에서 확장자 제거 및 특수 문자 제거
    file_root, file_ext = os.path.splitext(uploaded_file.name)
    file_root = re.sub(r'[^A-Za-z0-9_-]', '_', file_root)  # 특수 문자 제거

    # 업로드된 파일에서 텍스트 추출 및 데이터프레임 생성
    texts_df = create_texts_df([uploaded_file])  # 리스트로 감싸서 함수에 전달

    if texts_df.empty:
        st.error('업로드된 파일에서 텍스트를 추출할 수 없습니다.')
        st.stop()

    # 모델 학습 및 설정
    if model_selection not in MODELS:
        st.error("지원되지 않는 모델이 선택되었습니다.")
        st.stop()

    # 모델 옵션 적용
    if model_options.get('submit'):
        # 'submit' 키를 제외하고 다른 설정만 저장
        st.session_state['model_kwargs'] = {k: v for k, v in model_options.items() if k != 'submit'}
        st.success(f"{model_selection} 옵션이 적용되었습니다.")
        
        # 모델 학습 자동 실행
        with st.spinner('모델 학습 중...'):
            docs = generate_docs(texts_df, '단락', ngrams=st.session_state.ngrams)
            id2word, corpus, model = train_model(
                docs,
                MODELS[model_selection]['class'],
                **st.session_state.get('model_kwargs', {})
            )
        current_model = model  # 모델을 변수로 저장

        # 모델의 실제 토픽 수에 맞게 colors 초기화 및 저장
        if model.num_topics <= len(COLORS):
            st.session_state.colors = random.sample(COLORS, k=model.num_topics)
        else:
            # COLORS 리스트가 충분하지 않은 경우, 색상을 반복해서 사용
            st.session_state.colors = [COLORS[i % len(COLORS)] for i in range(model.num_topics)]

        st.session_state.id2word = id2word
        st.session_state.corpus = corpus

        st.success('모델 학습 완료!')

    else:
        st.info('모델 옵션을 설정하고 "적용" 버튼을 클릭하여 모델을 학습하세요.')
        st.stop()

    # 탭 설정
    tabs = st.tabs(["개별 파일 분석", "pyLDAvis 시각화"])

    with tabs[0]:
        st.header('개별 파일 분석')

        # 단일 파일이므로 루프 제거
        st.subheader(f'파일: {file_root}{file_ext}')
        file_df = texts_df[texts_df['파일명'] == uploaded_file.name]
        docs = generate_docs(file_df, '단락', ngrams=st.session_state.ngrams)

        # 1) 입력 문서 샘플
        with st.expander('입력 문서 샘플'):
            sample_size = min(5, len(docs))
            if sample_size > 0:
                sample_texts = random.sample(docs, sample_size)
                for idx, doc in enumerate(sample_texts):
                    st.markdown(f'**샘플 {idx + 1}**: _{" ".join(doc)}_')
            else:
                st.write("샘플 문서가 없습니다.")

        # 2) 워드 클라우드
        with st.expander('워드클라우드'):
            wc = generate_wordcloud(docs, collocations=st.session_state.collocations)
            st.image(wc.to_image(), caption='워드클라우드', use_column_width=True)

            # Download 버튼 추가
            buf = BytesIO()
            wc.to_image().save(buf, format='PNG')
            byte_im = buf.getvalue()
            st.download_button(
                label="워드클라우드 다운로드",
                data=byte_im,
                file_name=f'wordcloud_{file_root}.png',  # 확장자 제거된 파일 이름 사용
                mime='image/png'
            )

        # 3) 상위 20개 키워드 빈도수
        with st.expander('상위 20개 키워드 빈도수'):
            all_words = [word for doc in docs for word in doc]
            word_freq = pd.Series(all_words).value_counts().reset_index()
            word_freq.columns = ['키워드', '빈도수']
            top_20_words = word_freq.head(20)

            # Plotly Express를 사용한 인터랙티브 바 차트 생성
            fig = px.bar(
                top_20_words,
                x='키워드',
                y='빈도수',
                labels={'키워드': '키워드', '빈도수': '빈도수'},
                title='상위 20개 키워드 빈도수',
                text='빈도수',
                color='빈도수',
                color_continuous_scale='Blues'
            )

            # 레이아웃 업데이트 (옵션)
            fig.update_layout(
                xaxis_title='키워드',
                yaxis_title='빈도수',
                bargap=0.2,  # 막대 사이 간격
                template='plotly_white'  # 흰색 배경 템플릿
            )

            # 텍스트 레이블 포맷 설정
            fig.update_traces(texttemplate='%{text}', textposition='outside')

            # Plotly 바 차트를 Streamlit에 표시
            st.plotly_chart(fig, use_container_width=True)

            # 2회 이상 나타난 키워드 목록 및 엑셀 다운로드
            keywords_twice = word_freq[word_freq['빈도수'] >= 2]
            st.subheader('2회 이상 나타난 키워드 목록')

            # 데이터프레임을 양 옆으로 넓게 표시
            st.dataframe(keywords_twice, use_container_width=True)

            # 엑셀 파일로 다운로드 (수정된 부분)
            towrite = BytesIO()
            keywords_twice.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(
                label="엑셀 파일로 다운로드",
                data=towrite,
                file_name=f'keywords_twice_{file_root}.xlsx',  # 확장자 제거된 파일 이름 사용
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        # 4) 토픽 단어 가중치 요약
        with st.expander('토픽 단어 가중치 요약'):
            # 토픽 단어 가중치 요약 섹션
            num_topics = st.session_state['model_kwargs'].get('num_topics', 9)
            topics = current_model.show_topics(
                formatted=False,
                num_words=50,
                num_topics=num_topics,
                log=False
            )
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

        # 5) 토픽 키워드 워드클라우드
        with st.expander('토픽 키워드 워드클라우드'):
            colors = st.session_state.colors  # session_state에서 colors 참조

            def make_color_func(color):
                def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                    return color
                return color_func

            cols = st.columns(3)  # 3개의 열로 나누기
            for index, topic in enumerate(topics):
                # 색상을 선택
                color = colors[index % len(colors)]

                # color_func를 생성
                color_func_custom = make_color_func(color)

                # WordCloud 객체 생성
                wc = WordCloud(
                    font_path=WORDCLOUD_FONT_PATH,
                    width=1400,  # 고해상도
                    height=1200,  # 고해상도
                    background_color='white',
                    collocations=st.session_state.collocations,
                    prefer_horizontal=1.0,
                    color_func=color_func_custom  # 수정된 부분
                )

                # 토픽 단어 빈도수 딕셔너리 생성
                topic_freq = dict(topic[1])

                # 워드클라우드 생성
                wc.generate_from_frequencies(topic_freq)

                # 워드클라우드 이미지를 열에 표시
                with cols[index % 3]:
                    st.image(wc.to_image(), caption=f'토픽 #{index}', use_column_width=True)

                    # Download 버튼 추가
                    buf = BytesIO()
                    wc.to_image().save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    st.download_button(
                        label=f"토픽 {index} 워드클라우드 다운로드",
                        data=byte_im,
                        file_name=f'topic_{index}_wordcloud_{file_root}.png',  # 확장자 제거된 파일 이름 사용
                        mime='image/png'
                    )

        # 6) 토픽 하이라이트 문장
        with st.expander('토픽 하이라이트 문장'):
            for idx, doc in enumerate(docs[:10]):  # 샘플 크기 조정
                html_elements = []
                for token in doc:
                    if st.session_state.id2word.token2id.get(token) is None:
                        # 단어가 사전에 없으면 취소선 적용
                        html_elements.append(f'<span style="text-decoration:line-through;">{token}</span>')
                    else:
                        term_topics = current_model.get_term_topics(token, minimum_probability=0)
                        topic_probabilities = [term_topic[1] for term_topic in term_topics]
                        max_topic_probability = max(topic_probabilities) if topic_probabilities else 0
                        if max_topic_probability < st.session_state.highlight_probability_minimum:
                            # 확률이 최소값보다 낮으면 일반 텍스트
                            html_elements.append(token)
                        else:
                            # 최대 확률을 가진 토픽의 색상으로 하이라이트
                            max_topic_index = topic_probabilities.index(max_topic_probability)
                            max_topic = term_topics[max_topic_index]
                            
                            # max_topic[0]이 colors 리스트의 범위를 벗어나지 않도록 확인
                            if max_topic[0] < len(st.session_state.colors):
                                background_color = st.session_state.colors[max_topic[0]]
                            else:
                                background_color = 'grey'  # 기본 색상 또는 대체 색상

                            color = white_or_black_text(background_color)
                            html_elements.append(
                                f'<span style="background-color: {background_color}; color: {color}; opacity: 0.5;">{token}</span>'
                            )
                st.markdown(f'문서 #{idx + 1}: {" ".join(html_elements)}', unsafe_allow_html=True)

        # 7) 메트릭
        with st.expander('메트릭'):
            metrics_section(current_model, corpus)

    with tabs[1]:
        st.header('pyLDAvis 시각화')
        try:
            py_lda_vis_data = pyLDAvis.gensim_models.prepare(
                current_model,
                corpus,
                id2word
            )
            py_lda_vis_html = pyLDAvis.prepared_data_to_html(py_lda_vis_data)

            # Display pyLDAvis in a larger iframe
            st.components.v1.html(py_lda_vis_html, width=1600, height=900, scrolling=True)

            # Download pyLDAvis HTML
            buffer = BytesIO()
            buffer.write(py_lda_vis_html.encode('utf-8'))
            buffer.seek(0)
            st.download_button(
                label="pyLDAvis HTML 다운로드",
                data=buffer,
                file_name=f'pyLDAvis_{file_root}.html',  # 필요에 따라 file_root 포함
                mime='text/html'
            )

        except Exception as e:
            st.error(f"pyLDAvis 시각화 생성 중 오류 발생: {e}")

if __name__ == '__main__':
    main()
