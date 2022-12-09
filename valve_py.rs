use crate::{
    configure_and_or_load as configure_and_or_load_rs,
    get_compiled_datatype_conditions as get_compiled_datatype_conditions_rs,
    get_compiled_rule_conditions as get_compiled_rule_conditions_rs,
    get_parsed_structure_conditions as get_parsed_structure_conditions_rs,
    insert_new_row as insert_new_row_rs, update_row as update_row_rs,
    validate::get_matching_values as get_matching_values_rs,
    validate::validate_row as validate_row_rs, valve_grammar::StartParser,
};
use futures::executor::block_on;
use pyo3::prelude::{pyfunction, pymodule, wrap_pyfunction, PyModule, PyResult, Python};
use serde_json::Value as SerdeValue;
use sqlx::{
    any::{AnyConnectOptions, AnyKind, AnyPoolOptions},
    query as sqlx_query,
};
use std::str::FromStr;

/// Given a &str representing the location of a (sqlite or postgresql) database, return a String in
/// the form required to connect to it via sqlx.
fn get_connection_string(db_path: &str) -> String {
    if !db_path.starts_with("postgresql://") && !db_path.starts_with("sqlite://") {
        return format!("sqlite://{}", db_path);
    } else {
        return db_path.to_string();
    }
}

/// Given a path to a table table file (table.tsv), a directory in which to find/create a database:
/// configure the database using the configuration which can be looked up using the table table,
/// and optionally load it if the `load` flag is set to true. If the `verbose` flag is also set to
/// true, output progress messages while loading.
#[pyfunction]
fn configure_and_or_load(
    table_table: &str,
    db_path: &str,
    load: bool,
    verbose: bool,
) -> PyResult<String> {
    let config = block_on(configure_and_or_load_rs(table_table, db_path, load, verbose)).unwrap();
    Ok(config)
}

/// Given a config map represented as a JSON string, a directory containing the database, the table
/// name and column name from which to retrieve matching values, return a JSON array (represented as
/// a string) of possible valid values for the given column which contain the matching string as a
/// substring (or all of them if no matching string is given). The JSON array returned is formatted
/// for Typeahead, i.e., it takes the form: [{"id": id, "label": label, "order": order}, ...].
#[pyfunction]
fn get_matching_values(
    config: &str,
    db_path: &str,
    table_name: &str,
    column_name: &str,
    matching_string: Option<&str>,
) -> PyResult<String> {
    let config: SerdeValue = serde_json::from_str(config).unwrap();
    let config = config.as_object().unwrap();

    let connection_options =
        AnyConnectOptions::from_str(get_connection_string(db_path).as_str()).unwrap();
    let pool = AnyPoolOptions::new().max_connections(5).connect_with(connection_options);
    let pool = block_on(pool).unwrap();

    let parser = StartParser::new();
    let compiled_datatype_conditions = get_compiled_datatype_conditions_rs(&config, &parser);
    let parsed_structure_conditions = get_parsed_structure_conditions_rs(&config, &parser);

    let matching_values = block_on(get_matching_values_rs(
        &config,
        &compiled_datatype_conditions,
        &parsed_structure_conditions,
        &pool,
        table_name,
        column_name,
        matching_string,
    ))
    .unwrap();

    Ok(matching_values.to_string())
}

/// Given a config map represented as a JSON string, a directory in which to find the database,
/// a table name, a row, and if the row already exists in the database, its associated row number,
/// perform both intra- and inter-row validation and return the validated row as a JSON string.
#[pyfunction]
fn validate_row(
    config: &str,
    db_path: &str,
    table_name: &str,
    row: &str,
    existing_row: bool,
    row_number: Option<u32>,
) -> PyResult<String> {
    let config: SerdeValue = serde_json::from_str(config).unwrap();
    let config = config.as_object().unwrap();
    let row: SerdeValue = serde_json::from_str(row).unwrap();
    let row = row.as_object().unwrap();

    let connection_options =
        AnyConnectOptions::from_str(get_connection_string(db_path).as_str()).unwrap();
    let pool = AnyPoolOptions::new().max_connections(5).connect_with(connection_options);
    let pool = block_on(pool).unwrap();

    let parser = StartParser::new();
    let compiled_datatype_conditions = get_compiled_datatype_conditions_rs(&config, &parser);
    let compiled_rule_conditions =
        get_compiled_rule_conditions_rs(&config, compiled_datatype_conditions.clone(), &parser);

    let result_row = block_on(validate_row_rs(
        &config,
        &compiled_datatype_conditions,
        &compiled_rule_conditions,
        &pool,
        table_name,
        &row,
        existing_row,
        row_number,
    ))
    .unwrap();

    Ok(SerdeValue::Object(result_row).to_string())
}

/// Given a config map represented as a JSON string, a directory in which the database is located,
/// a table name, a row represented as a JSON string, and its associated row number, update the row
/// in the database.
#[pyfunction]
fn update_row(
    config: &str,
    db_path: &str,
    table_name: &str,
    row: &str,
    row_number: u32,
) -> PyResult<()> {
    let config: SerdeValue = serde_json::from_str(config).unwrap();
    let config = config.as_object().unwrap();
    let row: SerdeValue = serde_json::from_str(row).unwrap();
    let row = row.as_object().unwrap();

    let connection_options =
        AnyConnectOptions::from_str(get_connection_string(db_path).as_str()).unwrap();
    let pool = AnyPoolOptions::new().max_connections(5).connect_with(connection_options);
    let pool = block_on(pool).unwrap();
    if pool.any_kind() == AnyKind::Sqlite {
        block_on(sqlx_query("PRAGMA foreign_keys = ON").execute(&pool)).unwrap();
    }

    block_on(update_row_rs(&config, &pool, table_name, &row, row_number)).unwrap();

    Ok(())
}

/// Given a config map represented as a JSON string, a directory in which the database is located,
/// a table name, and a row represented as a JSON string, insert the new row to the database.
#[pyfunction]
fn insert_new_row(config: &str, db_path: &str, table_name: &str, row: &str) -> PyResult<u32> {
    let config: SerdeValue = serde_json::from_str(config).unwrap();
    let config = config.as_object().unwrap();
    let row: SerdeValue = serde_json::from_str(row).unwrap();
    let row = row.as_object().unwrap();

    let connection_options =
        AnyConnectOptions::from_str(get_connection_string(db_path).as_str()).unwrap();
    let pool = AnyPoolOptions::new().max_connections(5).connect_with(connection_options);
    let pool = block_on(pool).unwrap();
    if pool.any_kind() == AnyKind::Sqlite {
        block_on(sqlx_query("PRAGMA foreign_keys = ON").execute(&pool)).unwrap();
    }

    let new_row_number = block_on(insert_new_row_rs(&config, &pool, table_name, &row)).unwrap();
    Ok(new_row_number)
}

#[pymodule]
fn ontodev_valve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(configure_and_or_load, m)?)?;
    m.add_function(wrap_pyfunction!(get_matching_values, m)?)?;
    m.add_function(wrap_pyfunction!(validate_row, m)?)?;
    m.add_function(wrap_pyfunction!(update_row, m)?)?;
    m.add_function(wrap_pyfunction!(insert_new_row, m)?)?;
    Ok(())
}
