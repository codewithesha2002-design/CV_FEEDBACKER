import mysql.connector


def create_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="your_password",
        database="resume_analyzer",
    )
    return conn


def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            role VARCHAR(100),
            confidence FLOAT,
            decision VARCHAR(50)
        )
        """
    )

    conn.commit()
    cursor.close()
    conn.close()


def insert_candidate(name, role, confidence, decision):
    conn = create_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO candidates (name, role, confidence, decision)
    VALUES (%s, %s, %s, %s)
    """

    values = (name, role, confidence, decision)

    cursor.execute(query, values)
    conn.commit()

    cursor.close()
    conn.close()


def get_candidates():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM candidates")
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data
