import streamlit as st
import random
import utils

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.keypoint_classifier import recognition
from model.dynamic_classifier.dynamic_classifier import model_exists as dynamic_model_exists

DYNAMIC_LETTERS  = frozenset('ҐДЄЗЇЙКЦЩЬ')
SEQUENCE_FRAMES  = 15


def change_level(level):
    st.session_state.clear()
    st.session_state["level"] = level
    if level != "menu":
        reset_game()


def reset_game():
    levels = {
        "easy": (["ЛАМПА", "МЕТА", "СИЛА", "ЛИСТ", "ТЕПЛО", "ПАН", "СЕЛО", "МАТИ", "ПОЛЕ", "САЛО", "ЛОТО", "ТОН", "СТАН", "СМОЛА", "ЛИПА", "СИН", "НАСИП", "ЛОТОС",
                  "КІТ", "ЗУБ", "ДЕНЬ", "КОТ", "КАЗКА", "ЗИМА", "ЙОГА", "КИТ", "ДІМ", "ЛЕД"], 10),
        "medium": (["МІСТО", "ІСПИТ", "РОБОТА", "МОТИВ", "НЕБО", "МІСТ", "ВИСОТА", "СУМА", "ПЕРО", "ЧОРНИЛА", "ТІСТО", "СТІЛ", "ЛІТОПИС", "ВІТЕР", "ТУМАН", "ВЕЧІР", "ПОБУТ", "БОЛОТО", "ЛІТР", "СТОВП", "БЕТОН",
                    "КОЗАК", "ДЕРЕВО", "ЗЕРНО", "КОБРА", "ДВЕРІ", "ЗІРКА", "ЙОЛОП", "КІНЕЦЬ", "ДОЩ", "ЦИРК"], 10),
        "hard": (["УСПІХ", "ГУМОР", "ШИЯ", "ЮРИСТ", "ЧЕМПІОН", "СИМВОЛ", "ФАХ", "СПАЛАХ", "ІНЖЕНЕР", "ЛЮБОВ", "ПЕЧИВО", "ЛИСТЯ", "ФІЛОЛОГІЯ", "ФОРМА", "ГОРА", "ХВІСТ", "ФАНЕРА", "ШТАНИ", "СТРУМ",
                  "ДЕРЖАВА", "КОЗАЦТВО", "ЄДНІСТЬ", "ЗБРОЯРСТВО", "ЦІННІСТЬ", "ЩЕДРІСТЬ", "ДРУЖБА", "КУЛЬТУРА"], 10),
    }
    words, tries = levels[st.session_state["level"]]
    st.session_state["random_word"] = random.choice(words)
    st.session_state["count"] = tries
    st.session_state["guessed_letters"] = []
    st.session_state["not_guessed_letters"] = []
    st.session_state["recognized_letter"] = ""
    st.session_state["game_won"] = False
    st.session_state["display_word"] = " ".join(["_" for _ in st.session_state["random_word"]])
    st.session_state["dynamic_mode"] = False
    st.session_state["dynamic_buffer"] = []


def app():
    utils.load_css("style.css")

    if "level" not in st.session_state:
        st.session_state.level = "menu"

    # ── Level selection ───────────────────────────────────────────
    if st.session_state.level == "menu":
        st.markdown('''
        <div class="page-hero">
            <div class="title_header">Гра</div>
            <p>Оберіть рівень складності та починайте</p>
        </div>
        ''', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        level_meta = [
            (col1, "Легкий",   "Прості слова,\nідеально для початку",   "easy_button",   "easy"),
            (col2, "Середній", "Складніші слова,\nдля практики",         "medium_button", "medium"),
            (col3, "Складний", "Для досвідчених,\nрізноманітні слова",   "hard_button",   "hard"),
        ]

        for col, name, desc, key, level in level_meta:
            with col:
                st.markdown(f'''
                <div class="level-header">
                    <div class="level-name">{name}</div>
                    <div class="level-desc">{desc}</div>
                </div>
                ''', unsafe_allow_html=True)
                st.button("Грати", on_click=change_level, args=(level,), key=key, use_container_width=True)

    # ── Active game ───────────────────────────────────────────────
    else:
        st.session_state.easy = st.empty()
        st.session_state.medium = st.empty()
        st.session_state.hard = st.empty()

        level_titles = {
            "easy":   "Легкий рівень",
            "medium": "Середній рівень",
            "hard":   "Складний рівень",
        }

        images = [
            "images/10.10 (1).svg",
            "images/10.9 (1).svg",
            "images/10.8 (1).svg",
            "images/10.7 (1).svg",
            "images/10.6 (1).svg",
            "images/10.5 (1).svg",
            "images/10.4 (1).svg",
            "images/10.3 (1).svg",
            "images/10.2 (1).svg",
            "images/10.1 (1).svg",
        ]

        level = st.session_state.level
        st.markdown(f'<div class="title_subheader">{level_titles[level]}</div>', unsafe_allow_html=True)

        if "random_word" not in st.session_state:
            reset_game()

        count = st.session_state["count"]
        game_won = st.session_state.get("game_won", False)

        if "images" not in st.session_state:
            st.session_state.images = images

        img_index = max(0, min(len(images) - 1, len(images) - count))

        col1, col2 = st.columns(2)
        with col1:
            if game_won:
                st.image("images/hardwinn.svg", width=250)
            elif count == 0:
                st.image("images/loseee.svg", width=250)
            else:
                st.image(images[img_index], width=250)

        with col2:
            if not game_won and count > 0:
                if "camera_key" not in st.session_state:
                    st.session_state.camera_key = 0
                img_file = st.camera_input(
                    "Покажіть жест / Show your gesture",
                    key=f"camera_{st.session_state.camera_key}"
                )
                if img_file is not None:
                    letter, annotated_image = recognition.process_frame(img_file)
                    if letter:
                        st.session_state["recognized_letter"] = letter
                        recognition.process_letter()
                        st.session_state.camera_key += 1
                        st.rerun()
                    else:
                        st.warning("Руку не виявлено / No hand detected. Спробуйте ще / Try again.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        recognized = st.session_state.get("recognized_letter", "") or "—"
        display_word = st.session_state.get("display_word", "")
        guessed = st.session_state.get("guessed_letters", [])
        not_guessed = st.session_state.get("not_guessed_letters", [])

        guessed_html = "".join([f'<span class="chip">{l}</span>' for l in guessed]) if guessed else "<span style='color:#aaa;font-size:14px'>—</span>"
        missed_html = "".join([f'<span class="chip wrong">{l}</span>' for l in not_guessed]) if not_guessed else "<span style='color:#aaa;font-size:14px'>—</span>"

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'''
            <div class="game-stat">
                <div class="game-stat-label">Розпізнаний жест</div>
                <div class="game-stat-value">{recognized}</div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="game-stat">
                <div class="game-stat-label">Слово</div>
                <div class="game-stat-value">{display_word}</div>
            </div>
            ''', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'''
            <div class="game-stat">
                <div class="game-stat-label">Вгадані літери</div>
                <div style="padding-top:4px;">{guessed_html}</div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="game-stat">
                <div class="game-stat-label">Невгадані літери</div>
                <div style="padding-top:4px;">{missed_html}</div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
        st.button("Назад до меню", on_click=lambda: change_level("menu"), key="back_1button", use_container_width=True)
