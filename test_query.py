import urllib.request
import json

req = urllib.request.Request(
    'http://127.0.0.1:2002/voltForge-ai/api/v1/model/chat',
    headers={
        'X-Voltforge-AI-Token': 'vfa_sec_9f7a8b3c1d2e4f5a6b7c8d9e0f1a2b3c',
        'Content-Type': 'application/json'
    },
    data=json.dumps({
        'message': 'u edit them and fix the circuit in workspace now',
        'boardType': 'ARDUINO_UNO',
        'components': [
            {'id': 'node_just_relay', 'type': 'RELAY_SINGLE', 'name': '5V Relay Module'},
            {'id': 'node_just_motor_supply', 'type': 'DC_MOTOR', 'name': '5V DC Motor'}
        ],
        'wires': [
            {'from': 'node_just_relay:VCC', 'to': 'board:5V'},
            {'from': 'node_just_relay:GND', 'to': 'board:GND'}
        ],
        'history': [
            {'role': 'user', 'content': 'is the code & circuite correct?'},
            {'role': 'assistant', 'content': 'The circuit is missing digital pin connection to relay IN and flyback protection.'}
        ]
    }).encode('utf-8')
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    print("=== MODEL REPLY ===")
    print(res["reply"])
    print("=== PROPOSALS ===")
    print("Wires:", res.get("wireSuggestions"))
    print("HasCode:", res.get("hasCode"))
