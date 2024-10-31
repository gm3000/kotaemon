#!/usr/bin/env python3

import time
import os
import sys
from gradio_client import Client

client = Client("http://127.0.0.1:7860/")

history = []

settings = {
    'logged_in': False,
    'selected_file_id': None,
    'session_id': None
}

def login():
    user_name = input("User name: ")
    user_pass = input("Pass: ")
    client.predict(user_name, user_pass, api_name="/login")
    settings['logged_in'] = True
    print(f"Chat start")

def pick_file():
    resp = client.predict(api_name="/list_file_1")
    all_files = resp['data']
    all_files_option_test = '\n'.join([f"{index}. {value[1]} {value[2]} {value[5]}" for index, value in enumerate(all_files)])
    print('\n')
    print(all_files_option_test)
    user_pick = input("Pick one file: ")
    settings['selected_file_id'] = all_files[int(user_pick)][0]


def init_session():
    result_chat = client.predict(chat_input='blablabla', chat_history=None, conv_name=None, api_name="/submit_msg")
    settings['session_id'] = result_chat[2]['value']

def chat():
    while True:
        user_input = input("You: ")
        history.append(user_input)  
        if user_input.lower() == "/bye":
            print("Chat: Goodbye! Closing the application.")
            break
        
        # Here you can add more complex logic for generating responses
        response = client.predict([[user_input, None]], None, None, "select", [settings['selected_file_id']], api_name="/chat_fn")
        resp = response[0][0][1]
        print(f"Bot: {resp}")

def main():
    login()
    pick_file()
    init_session()
    chat()


if __name__ == "__main__":
    main()