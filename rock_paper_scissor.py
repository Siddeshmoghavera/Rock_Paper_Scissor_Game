import cv2
import mediapipe as mp
import random
import time
import os

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Choices
choices = ["rock", "paper", "scissors"]
score_player = 0
score_computer = 0
last_choice = None
countdown_start = time.time()
countdown_seconds = 3
playing = False

# Load computer choice images
img_folder = "images"
img_assets = {
    "rock": cv2.imread(os.path.join(img_folder, "rock.png")),
    "paper": cv2.imread(os.path.join(img_folder, "paper.png")),
    "scissors": cv2.imread(os.path.join(img_folder, "scissors.png"))
}

# Function to classify gesture
def classify_gesture(landmarks):
    if landmarks:
        tips_ids = [8, 12, 16, 20]
        fingers = []
        for tip in tips_ids:
            if landmarks[tip].y < landmarks[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        if fingers == [0, 0, 0, 0]:
            return "rock"
        elif fingers == [1, 1, 1, 1]:
            return "paper"
        elif fingers == [1, 1, 0, 0]:
            return "scissors"
    return None

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    player_choice = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            player_choice = classify_gesture(hand_landmarks.landmark)

    # Countdown logic
    elapsed = time.time() - countdown_start
    remaining = countdown_seconds - int(elapsed)

    if remaining > 0:
        # Show countdown number
        cv2.putText(frame, str(remaining), (250, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    4, (0, 255, 255), 6, cv2.LINE_AA)
    else:
        if not playing:
            # Computer makes a move
            computer_choice = random.choice(choices)
            last_choice = (player_choice if player_choice else "none", computer_choice)

            # Determine winner
            if player_choice and computer_choice:
                if player_choice == computer_choice:
                    winner = "Draw"
                elif (player_choice == "rock" and computer_choice == "scissors") or \
                     (player_choice == "paper" and computer_choice == "rock") or \
                     (player_choice == "scissors" and computer_choice == "paper"):
                    winner = "You Win!"
                    score_player += 1
                else:
                    winner = "Computer Wins!"
                    score_computer += 1
            playing = True
            show_result_time = time.time()

        # Show result for 2 seconds
        if playing and time.time() - show_result_time < 2:
            cv2.putText(frame, f"You: {last_choice[0]}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Score - You: {score_player}  Comp: {score_computer}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if last_choice:
                comp_img = img_assets[last_choice[1]]
                if comp_img is not None:
                    cv2.imshow("Computer's Choice", comp_img)
        elif playing:
            # Restart countdown
            countdown_start = time.time()
            playing = False

    cv2.imshow("Your Hand Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
