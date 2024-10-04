from pydantic import BaseModel, field_validator

class MusicItem(BaseModel):
    id: str
    title: str
    categories: str
    brand: str
    sales_type: str


    @field_validator("categories", mode="before")
    @classmethod
    def validate_categories(cls, v):
        for char in ["[", "]", "'"]:
            v = v.replace(char, "")
        return v

    @field_validator("id", mode="before")
    @classmethod
    def validate_ids(cls, v):
        v = str(v)
        while v[0] == "0":
            v = v[1:]
        return v

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"{self.id}::{self.title}::{self.brand}::{self.categories}::{self.sales_type}"