import csv
import os


TASKS = []
TODOS_FILE = "todos.csv"


def add_one_task(title):
	"""Create a task and append it to the in-memory task list."""
	if not isinstance(title, str):
		raise TypeError("title must be a string")

	clean_title = title.strip()
	if clean_title == "":
		raise ValueError("title cannot be empty")

	task = {"title": clean_title, "done": False}
	TASKS.append(task)
	return task


def print_list():
	"""Print all tasks with a clear 1-based numeric position."""
	if len(TASKS) == 0:
		print("No tasks yet.")
		return

	for index, task in enumerate(TASKS, start=1):
		status = "x" if task["done"] else " "
		print(f"{index}. [{status}] {task['title']}")


def delete_task(number_to_delete):
	"""Delete one task using its 1-based list position."""
	if not isinstance(number_to_delete, int):
		raise TypeError("number_to_delete must be an integer")

	if number_to_delete < 1 or number_to_delete > len(TASKS):
		raise ValueError("number_to_delete is out of range")

	return TASKS.pop(number_to_delete - 1)


def save_todos():
	"""Persist all tasks into todos.csv."""
	with open(TODOS_FILE, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=["title", "done"])
		writer.writeheader()
		for task in TASKS:
			writer.writerow({"title": task["title"], "done": task["done"]})


def load_todos():
	"""Load tasks from todos.csv into the in-memory TASKS list."""
	if not os.path.exists(TODOS_FILE):
		TASKS.clear()
		return TASKS

	loaded_tasks = []
	with open(TODOS_FILE, "r", newline="", encoding="utf-8") as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			title = (row.get("title") or "").strip()
			done_value = (row.get("done") or "").strip().lower()
			done = done_value in ("true", "1", "yes", "y")
			if title != "":
				loaded_tasks.append({"title": title, "done": done})

	TASKS.clear()
	TASKS.extend(loaded_tasks)
	return TASKS


def print_menu():
	"""Render the CLI menu options."""
	print("\nTodo App")
	print("1. Create task")
	print("2. List tasks")
	print("3. Delete task")
	print("4. Save tasks")
	print("5. Load tasks")
	print("6. Exit")


def run_cli():
	"""Run the interactive command-line todo application."""
	load_todos()

	while True:
		print_menu()
		choice = input("Choose an option (1-6): ").strip()

		if choice == "1":
			title = input("Task title: ").strip()
			try:
				add_one_task(title)
				print("Task added.")
			except (TypeError, ValueError) as error:
				print(f"Error: {error}")
		elif choice == "2":
			print_list()
		elif choice == "3":
			if len(TASKS) == 0:
				print("No tasks to delete.")
				continue

			print_list()
			number_text = input("Task number to delete: ").strip()
			try:
				number_to_delete = int(number_text)
				removed_task = delete_task(number_to_delete)
				print(f"Deleted: {removed_task['title']}")
			except ValueError:
				print("Error: please enter a valid task number in range.")
			
		elif choice == "4":
			save_todos()
			print("Tasks saved.")
		elif choice == "5":
			load_todos()
			print("Tasks loaded.")
		elif choice == "6":
			save_todos()
			print("Tasks saved. Goodbye.")
			break
		else:
			print("Invalid option. Please choose a number from 1 to 6.")


if __name__ == "__main__":
	run_cli()