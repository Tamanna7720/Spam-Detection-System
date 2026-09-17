from app import app, save_prediction


def test_app_predict_endpoint():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"message": "Congratulations! You won a free prize. Claim now!"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True


def test_save_prediction_function_exists():
    assert callable(save_prediction)
