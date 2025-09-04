# game.py

import random
from pathlib import Path

# Create a Path object for your folder

class HangmanGame:
    def __init__(self, word_list=None, max_attempts=5):
        if word_list is None:
            folder_path = Path('static/pokemon')
            self.word_list = [file.stem for file in folder_path.glob('*.png')]
        else:
            self.word_list = word_list
        
        self.max_attempts = max_attempts
        self.reset_game()

    def reset_game(self):
        self.word = random.choice(self.word_list).lower()
        self.guessed_letters = set()
        self.attempts_left = self.max_attempts
        self.status = "playing"  # "playing", "won", "lost"

    def guess_letter(self, letter):
        letter = letter.lower()
        if self.status != "playing":
            return self.get_game_state()

        if letter in self.guessed_letters:
            return self.get_game_state()

        self.guessed_letters.add(letter)

        if letter not in self.word:
            self.attempts_left -= 1

        if all(c in self.guessed_letters for c in self.word):
            self.status = "won"
        elif self.attempts_left <= 0:
            self.status = "lost"

        return self.get_game_state()

    def get_game_state(self):
        display_word = "".join(c if c in self.guessed_letters else "_" for c in self.word)
        return {
            "word": display_word,
            "guessed_letters": list(self.guessed_letters),
            "attempts_left": self.attempts_left,
            "status": self.status,
            "current_pokemon": self.word  # <-- Added so frontend can display the correct image
        }