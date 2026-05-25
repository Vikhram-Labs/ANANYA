from ananya.schemas.datasets import InstructionRecord, PretrainRecord, SourceMetadata, SourceType, LanguageCode


def test_pretrain_record():
    r = PretrainRecord(
        id="t1",
        text="Article 1 test",
        language=LanguageCode.EN,
        article=1,
        source=SourceMetadata(
            source_id="x",
            source_type=SourceType.CONSTITUTION,
            language=LanguageCode.EN,
        ),
    )
    assert r.article == 1


def test_instruction_record():
    r = InstructionRecord(
        id="i1",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        task="constitutional_qa",
        language=LanguageCode.HI,
        source=SourceMetadata(
            source_id="x",
            source_type=SourceType.CONSTITUTION,
            language=LanguageCode.HI,
        ),
    )
    assert len(r.messages) == 3
