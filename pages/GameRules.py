import streamlit as st
import os
import utils

def app():
    utils.load_css("style.css")

    if 'language' not in st.session_state:
        st.session_state.language = 'uk'

    # Language toggle
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    col_empty, col1, col2, col_empty2 = st.columns([1, 2, 2, 1])
    with col1:
        if st.button("Українська", use_container_width=True, key="ua_btn_about_final"):
            st.session_state.language = 'uk'
    with col2:
        if st.button("English", use_container_width=True, key="en_btn_about_final"):
            st.session_state.language = 'en'

    texts = {
        'uk': {
            'title': "Правила гри",
            'tagline': "Як грати в Heart in Gestures",
            'rules': [
                "Обери рівень складності. Гра побудована на жестах алфавіту української мови. Гравець має нескінченну кількість спроб.",
                "На екрані з'явиться вікно камери — виконуй жести, і система розпізнає їх за допомогою скелетної моделі руки.",
                "Жести розпізнаються <strong> незалежно від руки </strong>.",
                "Слово відображається на екрані (<strong>_</strong>). Гравець повинен відтворити кожну літер за хронологією.",
                "Щоб гра зарахувала відповідь, тримай руку в одному положенні <strong> після чого натисни Take a photo </strong>. Якщо жест вже розпізнано, він позначається як «already captured».",
                "Букви <strong>Ґ Д Є З Ї Й К Ц Щ Ь</strong> відсутні — вони нестатичні.",
                "Гра доступна лише <strong>українською мовою</strong>.",
            ]
        },
        'en': {
            'title': "Game Rules",
            'tagline': "How to play Heart in Gestures",
            'rules': [
                "Choose a difficulty level. The game uses gestures of the Ukrainian sign alphabet. Each level gives you <strong>10 attempts</strong> to guess the word.",
                "A camera window will appear — perform gestures and the system will recognise them using a hand skeleton model.",
                "All gestures are performed with the <strong>right hand</strong>.",
                "The word is shown as a series of blanks (<strong>_</strong>). Each unguessed letter stays as an underscore.",
                "Hold your hand still for <strong>5 seconds</strong> for the system to register your answer. Already-captured gestures are marked as «already captured».",
                "The letters <strong>Ґ Д Є З Ї Й К Ц Щ Ь</strong> are absent — they are not static signs.",
                "Each wrong answer removes a petal from the flower. Lose all petals — game over. Guess all letters — the flower blooms!",
                "The game is available in <strong>Ukrainian only</strong>.",
            ]
        }
    }

    lang = st.session_state.language
    t = texts[lang]

    # Hero
    st.markdown(f'''
    <div class="page-hero">
        <div class="title_header">{t["title"]}</div>
        <p style="color:#ffffff; font-family:'Nunito',sans-serif; font-size:16px; margin-top:8px; font-weight:600;">
            {t["tagline"]}
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Step cards
    for i, rule in enumerate(t['rules']):
        st.markdown(f'''
        <div class="step-card">
            <div class="step-number">{i + 1}</div>
            <div class="step-text">{rule}</div>
        </div>
        ''', unsafe_allow_html=True)

