FLEX_ROW = {
    'display': 'flex',
    'flexDirection': 'row',
    'alignItems': 'center',
    'gap': '10px',
    'justifyContent': 'space-between',
}

FLEX_CENTERING = {
    'display': 'flex',
    'flexDirection': 'column',
    'alignItems': 'center',
    'justifyContent': 'center',
}

BOLD = {'fontWeight': 'bold'}

PAD = {'padding': '5px'}

DEFAULT_FONT = {
    'fontFamily': 'Segoe UI, sans-serif',
    'fontSize': '14px',
}

HEADER_STYLE = FLEX_ROW | DEFAULT_FONT | {
    'backgroundColor': '#f8f9fa',
    'borderBottom': '1px solid #dee2e6',
    'gap': '20px',
    'padding': '10px 20px',
    'flexWrap': 'nowrap',
}

INPUT_STYLE = {
    'height': '36px',
    'border': '1px solid #ccc',
    'borderRadius': '4px',
    'padding': '0 10px',
}

BUTTON_STYLE = FLEX_ROW | {
    'height': '36px',
    'min-width': '110px',
    'backgroundColor': '#007bff',
    'color': 'white',
    'border': 'none',
    'borderRadius': '4px',
    'padding': '0 20px',
    'fontWeight': '600',
    'cursor': 'pointer',
    'boxShadow': '0 2px 2px rgba(0,0,0,0.1)',
}

TAB_STYLE = {
    'justifyContent': 'center',
    'borderBottom': '1px solid #dee2e6',
}

TAB_HEADER_STYLE = FLEX_CENTERING | {
    'padding': '0px',
    'font-size': '14px',
    'height': '35px',
    'borderBottom': '1px solid #dee2e6',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
}

TAB_DESCR_STYLE = {
    'padding': '10px 10px 0px',
    'margin-bottom': '0px',
}

######

DROPDOWN_CONTAINER_STYLE = {
    'position': 'relative',
    'display': 'inline-block',
}

WER_BUTTON_STYLE = {
    'height': '36px',
    'backgroundColor': '#e9ecef',
    'border': '1px solid #ced4da',
    'borderRadius': '4px',
    'padding': '0 15px',
    'cursor': 'pointer',
    'fontWeight': 'bold',
    'color': '#495057',
    'whiteSpace': 'nowrap',
}

WER_CONTENT_STYLE = {
    'position': 'absolute',
    'top': '100%',
    'right': '0',
    'marginTop': '5px',
    'backgroundColor': '#fff',
    'border': '1px solid #ced4da',
    'borderRadius': '6px',
    'padding': '15px',
    'zIndex': '1000',
    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
    'display': 'none', # hidden by default
    'flexDirection': 'column',
    'gap': '15px',
}

WER_CHECKBOX = {
    'gap': '5px',
    'justifyContent': 'flex-start',
    'whiteSpace': 'nowrap',
}

