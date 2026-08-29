from google import genai


def main():
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=(
            "Reply with exactly this text and nothing else: "
            "TrustLab LLM connection works"
        ),
    )

    print(response.text)


if __name__ == "__main__":
    main()
