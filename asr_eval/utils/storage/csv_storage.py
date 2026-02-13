from collections.abc import Iterator
import csv
import json
from pathlib import Path
from typing import Any, override
import warnings

import polars as pl

from asr_eval.utils.serializing import deserialize_object, serialize_object
from asr_eval.utils.storage.base_storage import VALUE, BaseStorage


class CSVStorage(BaseStorage):
    """A csv-based :code:`BaseStorage` implementation.
    
    Note that :code:`BaseStorage` can use int/float/str/bool as key
    types.

    Warning:
        Gemini 3.0 LLM code!

    Note:
        While :code:`BaseStorage` interface is flexible and values
        can be of any pickleable type, CSV format is very limited
        and not typed. In this implementation, we try to serialize
        objects such as timed text segments into json and back, but this
        may cause unexpected behaviour or simply not work in some cases.
        Also note that row deletion/modification operation is extremely
        inefficient since it requires to rewrite the whole file.
        Finally, not that simultaneous modifications to the same file
        should not be done, or this may cause errors.
    """

    # The large code size is due to the BaseStorage requirements:
    # to be able to add new columns (that changes shema), and to delete
    # specific rows. Both actions require a full file rewriting.

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def parse_mixed(val: str) -> float | str:
            try:
                return float(val)
            except (ValueError, TypeError):
                return val
        
        # Load existing data or initialize empty state
        if self.path.exists() and self.path.stat().st_size > 0:
            self._df = pl.read_csv(
                self.path, schema_overrides={'value': pl.String}
            )
            self._df = self._df.with_columns(pl.col('value').map_elements(
                parse_mixed, return_dtype=pl.Object
            ))
        else:
            self._df = pl.DataFrame(schema_overrides={'value': pl.Object})

    # def _check_csv_compability(self, value: Any):
    #     """Ensures the value is naturally compatible with CSV (flat types)."""
    #     valid_types = (str, int, float, bool)
    #     if value is not None and not isinstance(value, valid_types):
    #         raise ValueError(
    #             f"CsvStorage only supports {valid_types}"
    #             f" for 'value', got {type(value)}"
    #         )

    MAGIC = '__json__!'

    @classmethod
    def _maybe_serialize(cls, value: Any) -> Any:
        valid_types = (str, int, float, bool)
        if value is not None and not isinstance(value, valid_types):
            # raise ValueError(
            #     f"CsvStorage only supports str, int, float, bool"
            #     f" for 'value', got {type(value)}"
            # )
            value = json.dumps(serialize_object(value), ensure_ascii=False)
            value = cls.MAGIC + value
        return value
    
    @classmethod
    def _maybe_deserialize(cls, value: Any) -> Any:
        if isinstance(value, str) and value.startswith(cls.MAGIC):
            value = value[len(cls.MAGIC):]
            value = deserialize_object(json.loads(value))
        return value

    def _get_exact_match_mask(self, **keys: VALUE) -> pl.Expr:
        """
        Creates a Polars expression that matches a row exactly.
        
        A row matches exactly if:
        1. All provided keys match the column values.
        2. All columns NOT provided in keys are Null.
        """

        conditions: list[pl.Expr] = []
        
        for k, v in keys.items():
            if k not in self._df.columns:
                return pl.lit(False)
            # FIX: Ensure comparison calculates to strictly True or False
            conditions.append((pl.col(k) == v).fill_null(False))
            
        for col in self._df.columns:
            if col != 'value' and col not in keys:
                conditions.append(pl.col(col).is_null())
        
        if not conditions:
            return pl.lit(True)
            
        expr = conditions[0]
        for cond in conditions[1:]:
            expr = expr & cond
        return expr

    @override
    def has_row(self, **keys: VALUE) -> bool:
        if self._df.is_empty():
            return False
        
        mask = self._get_exact_match_mask(**keys)
        # Using select + item is generally faster than filtering whole frame
        return self._df.select(mask.any()).item()

    def _write_csv(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = self._df.with_columns(
                pl.col('value')
                .map_elements(str, return_dtype=pl.String)
            )
        df.write_csv(self.path)

    @override
    def add_row(self, value: Any, overwrite: bool = True, **keys: VALUE):
        # self._check_csv_compability(value)
        value = self._maybe_serialize(value)
        
        # Prepare the new row data
        new_row_dict = {**keys, 'value': value}
        
        # Check for existence logic
        exists = self.has_row(**keys)
        
        if exists:
            if not overwrite:
                raise ValueError(f'The key {keys} already exists')
            
            # If overwriting, we MUST rewrite the whole file because 
            # CSVs do not support random access updates.
            mask = self._get_exact_match_mask(**keys)
            
            # Start with all rows except the one being overwritten
            new_df = self._df.filter(~mask)

            # Create a 1-row DF for the new data. 
            # We use diagonal concat to handle schema alignment.
            row_df = pl.DataFrame([new_row_dict], schema_overrides={'value': pl.Object})
            self._df = pl.concat([new_df, row_df], how="diagonal")
            
            # Rewrite file
            self._write_csv()
            
        else:
            # Check if this creates new columns (schema evolution)
            # If we add a column, we can't simple 'append' to the text file
            # without rewriting the header and filling previous rows
            # with commas.
            current_cols = set(self._df.columns)
            new_cols = set(new_row_dict.keys())
            
            schema_changed = not new_cols.issubset(current_cols)
            
            # Update memory
            row_df = pl.DataFrame([new_row_dict], schema_overrides={'value': pl.Object})
            
            if self._df.is_empty():
                self._df = row_df
                # First write
                self._write_csv()
            else:
                self._df = pl.concat([self._df, row_df], how="diagonal")
                
                if schema_changed:
                    # New columns introduced -> Rewrite full
                    # file to update header
                    self._write_csv()
                else:
                    # Schema matches -> Efficient Append
                    # We use csv standard lib to append strictly
                    # to avoid overhead
                    # Ensure the dict follows the dataframe column order
                    with open(self.path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=self._df.columns)
                        # We only write the row dict.
                        # Polars handles Nones -> "" (empty) 
                        # logic internally, but csv.DictWriter needs explicit
                        # None handling usually.
                        # However, pl.DataFrame structure ensures alignment.
                        # We pass the row dict ensuring we fill missing
                        # keys with None relative to schema
                        append_data = {
                            col: new_row_dict.get(col, None)
                            for col in self._df.columns
                        }
                        writer.writerow(append_data)

    @override
    def get_row(self, **keys: VALUE) -> Any:
        if self._df.is_empty():
            raise KeyError(str(keys))

        mask = self._get_exact_match_mask(**keys)
        filtered = self._df.filter(mask)
        
        if filtered.height == 0:
            raise KeyError(str(keys))
        
        # Return the 'value' column of the first (and only) match
        result = self._maybe_deserialize(filtered['value'].item())

        return result

    @override
    def delete_row(self, missing_ok: bool = False, **keys: VALUE):
        if self._df.is_empty():
            if missing_ok:
                return
            raise KeyError(str(keys))

        mask = self._get_exact_match_mask(**keys)
        
        # Check if exists (computationally cheaper to check count first
        # if strict)
        if not missing_ok:
            if not self._df.select(mask.any()).item():
                raise KeyError(str(keys))
        
        # Keep everything that DOES NOT match the mask
        self._df = self._df.filter(~mask)
        
        # Rewrite file
        self._write_csv()

    @override
    def list_all(
        self,
        load_values: bool = False,
        **keys: VALUE,
    ) -> pl.DataFrame:
        if self._df.is_empty():
            # Return empty schema-compliant or completely empty DF?
            # BaseStorage implies returning what we have.
            return pl.DataFrame(schema_overrides={'value': pl.Object})

        # Filter logic for list_all is "Subset Match":
        # Rows where specified keys match, ignoring omitted keys (wildcards).
        
        conditions: list[pl.Expr] = []
        for k, v in keys.items():
            if k not in self._df.columns:
                # If querying a column that doesn't exist, result is empty
                return pl.DataFrame(schema_overrides=self._df.schema)
            conditions.append(pl.col(k) == v)
            
        filtered_df = self._df
        if conditions:
            expr = conditions[0]
            for c in conditions[1:]:
                expr = expr & c
            filtered_df = self._df.filter(expr)

        # "Drops full-None columns"
        # We iterate columns and drop those that are all null in th
        # filtered result
        # Exception: Do not drop 'value' even if all null? 
        # The docstring says "Drops full-None columns", implying
        # implicit keys cleanup.
        valid_cols: list[str] = []
        for col_name in filtered_df.columns:
            if not filtered_df[col_name].is_null().all():
                valid_cols.append(col_name)
        
        # Ensure 'value' is kept if load_values is True,
        # though docstring logic 
        # implies removing null columns. Usually 'value'
        # isn't a key column so it stays 
        # if data exists.
        
        result = filtered_df.select(valid_cols)
        
        if not load_values and 'value' in result.columns:
            result = result.drop('value')
        elif 'value' in result.columns:
            result = result.with_columns(
                pl.col('value')
                .map_elements(self._maybe_deserialize, return_dtype=pl.Object)
                .alias('value')
            )
            print(result['value'])
            
        return result

    @override
    def iter_rows(
        self,
        load_values: bool = False,
        **keys: VALUE,
    ) -> Iterator[dict[str, Any]]:
        
        df_res = self.list_all(load_values=load_values, **keys)
        
        # Convert to python dicts. 
        # Note: Polars to_dicts returns None for nulls, which
        # matches requirements.
        for row in df_res.to_dicts():
            # Filter out None values to represent "not set" (sparse dict)
            # BaseStorage docstring: "Fills the 'not set' values with None"
            # referencing list_all. For iter_rows, it usually implies returning
            # the clean dictionary.
            yield row

    @override
    def delete_all(self, **keys: VALUE):
        if self._df.is_empty():
            return

        conditions: list[pl.Expr] = []
        for k, v in keys.items():
            if k not in self._df.columns:
                # Key doesn't exist, so no rows match these criteria.
                return
            
            # FIX: Convert Null comparisons to False.
            # (Null == 2) -> Null. We want it to be False
            # so that ~False becomes True (Keep).
            conditions.append((pl.col(k) == v).fill_null(False))
            
        if not conditions:
            # delete_all() with no keys clears everything
            self._df = self._df.clear()
        else:
            expr = conditions[0]
            for c in conditions[1:]:
                expr = expr & c
            
            # Keep rows that DO NOT match the query
            self._df = self._df.filter(~expr)
            
        self._write_csv()

    @override
    def close(self):
        # No open file handles are maintained persistently.
        pass