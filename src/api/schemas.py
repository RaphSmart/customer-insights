from pydantic import BaseModel
from typing import List

class SentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    label: str
    probabilities: List[float]
