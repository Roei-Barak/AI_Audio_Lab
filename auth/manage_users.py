#!/usr/bin/env python3
"""
CLI לניהול משתמשים — הרץ ישירות:
  python auth/manage_users.py
"""
import getpass
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from auth.auth import (
    add_user, remove_user, deactivate_user, activate_user,
    change_password, list_users,
)

SEP = "─" * 48


def print_users():
    users = list_users()
    print(f"\n{'משתמש':<20} {'סטטוס':<10} {'נוסף':<18} {'הערה'}")
    print(SEP)
    if not users:
        print("  אין משתמשים רשומים")
    for name, active, added, note in users:
        status = "✅ פעיל" if active else "🔴 חסום"
        print(f"  {name:<18} {status:<10} {added:<18} {note}")
    print()


def prompt_password(label="סיסמה") -> str:
    while True:
        pw1 = getpass.getpass(f"  {label}: ")
        pw2 = getpass.getpass("  אשר סיסמה: ")
        if pw1 == pw2:
            return pw1
        print("  ❌ הסיסמאות אינן תואמות, נסה שוב.")


def menu():
    while True:
        print(f"\n{'='*48}")
        print("   AI_Audio_Lab — ניהול משתמשים")
        print(f"{'='*48}")
        print("  [1]  הצג משתמשים")
        print("  [2]  הוסף משתמש")
        print("  [3]  הסר משתמש")
        print("  [4]  חסום משתמש (מניעת גישה)")
        print("  [5]  שחרר חסימה")
        print("  [6]  שנה סיסמה")
        print("  [0]  יציאה")
        print(f"{'-'*48}")
        choice = input("בחר: ").strip()

        if choice == "1":
            print_users()

        elif choice == "2":
            name = input("  שם משתמש: ").strip()
            if not name:
                continue
            pw = prompt_password()
            note = input("  הערה (רשות): ").strip()
            add_user(name, pw, note)
            print(f"  ✅ משתמש '{name}' נוסף")

        elif choice == "3":
            print_users()
            name = input("  שם משתמש להסרה: ").strip()
            confirm = input(f"  בטוח? [{name}] (כן/לא): ").strip().lower()
            if confirm in ("כן", "y", "yes"):
                if remove_user(name):
                    print(f"  ✅ '{name}' הוסר")
                else:
                    print(f"  ❌ משתמש לא נמצא")

        elif choice == "4":
            print_users()
            name = input("  שם משתמש לחסימה: ").strip()
            if deactivate_user(name):
                print(f"  🔴 '{name}' נחסם")
            else:
                print(f"  ❌ משתמש לא נמצא")

        elif choice == "5":
            print_users()
            name = input("  שם משתמש לשחרור: ").strip()
            if activate_user(name):
                print(f"  ✅ '{name}' שוחרר")
            else:
                print(f"  ❌ משתמש לא נמצא")

        elif choice == "6":
            print_users()
            name = input("  שם משתמש: ").strip()
            pw = prompt_password("סיסמה חדשה")
            if change_password(name, pw):
                print(f"  ✅ סיסמת '{name}' עודכנה")
            else:
                print(f"  ❌ משתמש לא נמצא")

        elif choice == "0":
            print("  להתראות!")
            break


if __name__ == "__main__":
    # CLI quick args: add / remove / list
    if len(sys.argv) >= 2:
        cmd = sys.argv[1]
        if cmd == "list":
            print_users()
        elif cmd == "add" and len(sys.argv) >= 4:
            add_user(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
            print(f"✅ {sys.argv[2]} נוסף")
        elif cmd == "remove" and len(sys.argv) >= 3:
            remove_user(sys.argv[2])
            print(f"✅ {sys.argv[2]} הוסר")
        elif cmd == "block" and len(sys.argv) >= 3:
            deactivate_user(sys.argv[2])
            print(f"🔴 {sys.argv[2]} נחסם")
        else:
            print("שימוש: python auth/manage_users.py [list|add <user> <pass>|remove <user>|block <user>]")
    else:
        menu()
