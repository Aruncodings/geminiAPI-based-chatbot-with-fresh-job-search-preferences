# AI Chatbot with Job Search (Flask + RapidAPI)

## 📌 Project Overview

This project is a **Flask-based AI Chatbot** integrated with **RapidAPI job search APIs**. It allows users to chat with an AI assistant and search for jobs directly from the same platform. The project is designed to be beginner-friendly yet production-ready, making it an excellent portfolio project.

## 🚀 Features

* **AI Chatbot Interface** – Users can chat with the AI directly on the website.
* **Job Search Integration** – Fetches job listings from RapidAPI and displays them dynamically.
* **User Authentication** – Includes login and registration pages for personalized experience.
* **Responsive UI** – Simple and clean HTML templates for better user experience.
* **Flask Backend** – Handles routes, sessions, and API requests efficiently.

## 🗂️ Folder Structure

```
ai_chatbot_project/
│
├── main.py                # Main Flask app entry point
├── job.py                 # Job search logic & RapidAPI integration
│
├── templates/             # HTML Templates for UI
│   ├── chat.html          # Chatbot interface
│   ├── home.html          # Homepage
│   ├── jobs.html          # Job search results page
│   ├── login.html         # User login page
│   └── register.html      # User registration page
```

## 🔧 Installation & Setup

Follow these steps to run the project locally:

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/ai-chatbot-rapidapi.git
cd ai-chatbot-rapidapi
```

2. **Create Virtual Environment & Install Dependencies**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup API Key**

* Get your free API key from [RapidAPI](https://rapidapi.com/).
* Open `job.py` and replace `YOUR_RAPID_API_KEY` with your actual key.

4. **Run the Flask App**

```bash
python main.py
```

Then open your browser and visit: `http://127.0.0.1:5000/`

## 💻 Usage

1. Visit the homepage (`home.html`).
2. Log in or register as a new user.
3. Navigate to the chatbot page and start chatting.
4. Use the job search page to find jobs by keyword.

## 🖼️ Screenshots

> *(Replace these placeholders with actual screenshots)*

* **Home Page**
* **Chat Interface**
* **Job Search Results**

## 🔑 Tech Stack

* **Backend:** Flask (Python)
* **Frontend:** HTML, CSS (Basic)
* **API:** RapidAPI (Job Search API)

## 📌 Future Improvements

* Add a database (SQLite or PostgreSQL) to store user chat history.
* Improve chatbot response quality using an AI API (OpenAI, Gemini, etc.).
* Add filters (location, salary range) for job search results.
* Make UI fully responsive with Bootstrap or Tailwind.

## 🤝 Contribution

Contributions are welcome! If you’d like to add features, fix bugs, or improve documentation:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Submit a Pull Request

## 📜 License

This project is licensed under the MIT License - feel free to use it in your own projects.
