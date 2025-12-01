from typing import TypedDict,Optional,Annotated

class PostState(TypedDict):
    input_json: Annotated[list[dict],"multiplex"]
    ner_result: Optional[dict]
    sentiment_result: Optional[str]
    translation_result: Optional[str]
    summary_result: Optional[str]
    combined_result: Optional[dict]
