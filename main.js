document.getElementById("generateBtn").addEventListener("click", function () {
    const question = document.getElementById("questionInput").value.trim();

    if (!question) {
        alert("질문을 입력해주세요!");
        return;
    }

    document.getElementById("answerText").textContent = "답변을 생성 중입니다...";

    fetch("http://localhost:7001/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("answerText").textContent = data.answer;
    })
    .catch(error => {
        document.getElementById("answerText").textContent = "⚠️ 서버 오류 발생!";
        console.error(error);
    });
});
