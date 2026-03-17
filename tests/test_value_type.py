"""Tests for the Variable value_type field."""

from __future__ import annotations

import pytest

from app.models.world_model import Variable, WorldModel


class TestVariableValueType:
    """Tests for value_type and unit fields on Variable."""

    def test_quantitative_variable(self):
        v = Variable(
            name="market_share",
            current_value="26%",
            value_type="quantitative",
            unit="%",
            description="Market share percentage",
            source_ref=["chunk_0"],
        )
        assert v.value_type == "quantitative"
        assert v.unit == "%"

    def test_qualitative_variable(self):
        v = Variable(
            name="consumer_sentiment",
            current_value="positive",
            value_type="qualitative",
            unit=None,
            description="Overall consumer sentiment",
            source_ref=["chunk_0"],
        )
        assert v.value_type == "qualitative"
        assert v.unit is None

    def test_quantitative_without_unit(self):
        """Quantitative variables can omit unit (defaults to None)."""
        v = Variable(
            name="revenue",
            current_value="100",
            value_type="quantitative",
            description="Revenue figure",
            source_ref=["chunk_0"],
        )
        assert v.value_type == "quantitative"
        assert v.unit is None

    def test_value_type_required(self):
        """value_type is a required field."""
        with pytest.raises(Exception):
            Variable(
                name="test",
                current_value="x",
                description="test",
                source_ref=[],
            )

    def test_variable_serialization(self):
        """Verify value_type and unit survive round-trip serialization."""
        v = Variable(
            name="price",
            current_value="2.5",
            value_type="quantitative",
            unit="yuan/bottle",
            description="Retail price",
            source_ref=["c1"],
        )
        data = v.model_dump()
        assert data["value_type"] == "quantitative"
        assert data["unit"] == "yuan/bottle"

        v2 = Variable.model_validate(data)
        assert v2.value_type == "quantitative"
        assert v2.unit == "yuan/bottle"

    def test_variable_in_world_model(self):
        """Variables with value_type work correctly inside WorldModel."""
        wm = WorldModel(
            actors=[],
            relationships=[],
            timeline=[],
            variables=[
                Variable(
                    name="market_share",
                    current_value="26%",
                    value_type="quantitative",
                    unit="%",
                    description="Market share",
                    source_ref=["c1"],
                ),
                Variable(
                    name="brand_perception",
                    current_value="strong",
                    value_type="qualitative",
                    unit=None,
                    description="Brand perception",
                    source_ref=["c2"],
                ),
            ],
            question="Test?",
        )
        assert len(wm.variables) == 2
        assert wm.variables[0].value_type == "quantitative"
        assert wm.variables[0].unit == "%"
        assert wm.variables[1].value_type == "qualitative"
        assert wm.variables[1].unit is None

    def test_mock_extraction_has_value_type(self):
        """Verify the mock LLM response includes value_type."""
        from app.services.extractor import _MOCK_RESPONSE
        for var in _MOCK_RESPONSE["variables"]:
            assert "value_type" in var, f"Mock variable {var['name']} missing value_type"
            assert var["value_type"] in ("quantitative", "qualitative")
