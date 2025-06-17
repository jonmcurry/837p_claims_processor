# Defines the callbacks for the Dash application
from dash.dependencies import Input, Output, State
from dash import html # Updated import
from dash import dcc # Added dcc import, might be used implicitly or for future components
from dash import dash_table # dash_table already imported correctly
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from claims_ui.app import app
from claims_ui.app.queries import claims_queries, metrics_queries
from claims_ui.app.layouts.processed_claims_layout import create_processed_claims_layout
from claims_ui.app.layouts.failed_claims_layout import create_failed_claims_layout
from claims_ui.app.layouts.processing_metrics_layout import create_processing_metrics_layout
from claims_ui.app.layouts.healthcare_analytics_layout import create_healthcare_analytics_layout

@app.callback(
    Output('main-content-area', 'children'),
    [Input('main-tabs', 'value')]
)
def render_tab_content(tab_value):
    if tab_value == 'tab-processed-claims':
        return create_processed_claims_layout()
    elif tab_value == 'tab-failed-claims':
        return create_failed_claims_layout()
    elif tab_value == 'tab-processing-metrics':
        return create_processing_metrics_layout()
    elif tab_value == 'tab-healthcare-analytics':
        return create_healthcare_analytics_layout()
    return html.H3('Select a tab')

# --- Callbacks for Processed Claims Tab ---
@app.callback(
    [Output('facility-filter-pc', 'options'),
     Output('payer-filter-pc', 'options')],
    [Input('main-tabs', 'value')]
)
def update_processed_claims_filters_options(tab_value):
    if tab_value == 'tab-processed-claims':
        try:
            return claims_queries.get_distinct_facilities(), claims_queries.get_distinct_payers()
        except Exception as e: print(f"Error populating PC dropdowns: {e}"); return [], []
    return [], []

@app.callback(
    [Output('processed-claims-table', 'data'),
     Output('processed-claims-table', 'columns')],
    [Input('apply-filters-pc-button', 'n_clicks')],
    [State('facility-filter-pc', 'value'), State('payer-filter-pc', 'value'),
     State('date-picker-range-pc', 'start_date'), State('date-picker-range-pc', 'end_date')]
)
def update_processed_claims_table(n_clicks, facility_values, payer_values, start_date, end_date):
    if n_clicks == 0: return [], []
    try:
        df = claims_queries.get_processed_claims(
            facility_id=facility_values, payer_id=payer_values,
            start_date=start_date, end_date=end_date, limit=200)
        if df.empty: return [], []
        columns = [{'name': i.replace('_', ' ').title(), 'id': i} for i in df.columns]
        return df.to_dict('records'), columns
    except Exception as e: print(f"Error updating PC table: {e}"); return [], []

@app.callback(
    Output('selected-claim-details-pc', 'children'),
    [Input('processed-claims-table', 'selected_rows')],
    [State('processed-claims-table', 'data')]
)
def display_selected_claim_details_pc(selected_rows, table_data):
    if selected_rows and table_data:
        try:
            selected_data = table_data[selected_rows[0]]
            return [html.H5("Selected Claim Details")] + [html.P(f"{k.replace('_', ' ').title()}: {v}") for k, v in selected_data.items()]
        except Exception as e: print(f"Error PC details: {e}"); return html.P("Error loading details.")
    return html.P("Select a row for details.")

# --- Callbacks for Failed Claims Tab ---
@app.callback(
    [Output('facility-filter-fc', 'options'),
     Output('failure-category-filter-fc', 'options')],
    [Input('main-tabs', 'value')]
)
def update_failed_claims_filters_options(tab_value):
    if tab_value == 'tab-failed-claims':
        try:
            return claims_queries.get_distinct_facilities(), claims_queries.get_distinct_failure_categories()
        except Exception as e: print(f"Error populating FC dropdowns: {e}"); return [], []
    return [], []

@app.callback(
    [Output('failed-claims-table', 'data'),
     Output('failed-claims-table', 'columns'),
     Output('failure-reason-bargraph-fc', 'figure')],
    [Input('apply-filters-fc-button', 'n_clicks')],
    [State('facility-filter-fc', 'value'), State('failure-category-filter-fc', 'value'),
     State('date-picker-range-fc', 'start_date'), State('date-picker-range-fc', 'end_date')]
)
def update_failed_claims_table_and_graph(n_clicks, facility_values, failure_category_values, start_date, end_date):
    if n_clicks == 0: return [], [], {'layout': {'title': {'text': 'Failure Analysis', 'x':0.5}}}
    try:
        df_failed = claims_queries.get_failed_claims(
            facility_id=facility_values, failure_category=failure_category_values,
            start_date=start_date, end_date=end_date, limit=200)

        fig = {'layout': {'title': {'text': 'Failure Analysis: No data for graph', 'x':0.5}}}
        if df_failed.empty: return [], [], fig

        columns_fc = [{'name': i.replace('_', ' ').title(), 'id': i} for i in df_failed.columns]
        data_fc = df_failed.to_dict('records')

        if 'failure_category' in df_failed.columns and not df_failed['failure_category'].dropna().empty:
            counts = df_failed['failure_category'].value_counts().reset_index()
            counts.columns = ['category', 'count']
            fig = px.bar(counts, x='category', y='count', title="Failure Counts by Category", labels={'category': 'Failure Category'})
        elif 'failure_reason' in df_failed.columns and not df_failed['failure_reason'].dropna().empty:
            counts = df_failed['failure_reason'].value_counts().reset_index()
            counts.columns = ['reason', 'count']
            fig = px.bar(counts, x='reason', y='count', title="Failure Counts by Reason", labels={'reason': 'Failure Reason'})
            if len(counts) > 10: fig.update_xaxes(tickangle=-45) # Improve readability for many reasons
        fig.update_layout(title_x=0.5)
        return data_fc, columns_fc, fig
    except Exception as e: print(f"Error FC table/graph: {e}"); return [], [], {'layout': {'title': {'text': 'Error loading graph', 'x':0.5}}}

@app.callback(
    Output('selected-failed-claim-details-fc', 'children'),
    [Input('failed-claims-table', 'selected_rows')],
    [State('failed-claims-table', 'data')]
)
def display_selected_failed_claim_details_fc(selected_rows, table_data):
    if selected_rows and table_data:
        try:
            selected_data = table_data[selected_rows[0]]
            return [html.H5("Selected Failed Claim Details")] + [html.P(f"{k.replace('_', ' ').title()}: {v}") for k, v in selected_data.items()]
        except Exception as e: print(f"Error FC details: {e}"); return html.P("Error loading details.")
    return html.P("Select a row for details.")

# --- Callbacks for Processing Metrics Tab ---
@app.callback(
    [Output('org-filter-pm', 'options'), Output('region-filter-pm', 'options'), Output('facility-filter-pm', 'options')],
    [Input('main-tabs', 'value')]
)
def update_processing_metrics_filters_options(tab_value):
    if tab_value == 'tab-processing-metrics':
        try:
            return metrics_queries.get_distinct_orgs(), metrics_queries.get_distinct_regions(), claims_queries.get_distinct_facilities()
        except Exception as e: print(f"Error PM dropdowns: {e}"); return [], [], []
    return [], [], []

@app.callback(
    [Output('kpi-summary-pm', 'children'), Output('daily-trends-graph-pm', 'figure'),
     Output('facility-comparison-graph-pm', 'figure'), Output('org-region-summary-graph-pm', 'figure')],
    [Input('apply-filters-pm-button', 'n_clicks')],
    [State('org-filter-pm', 'value'), State('region-filter-pm', 'value'), State('facility-filter-pm', 'value'),
     State('date-picker-range-pm', 'start_date'), State('date-picker-range-pm', 'end_date')]
)
def update_processing_metrics_graphs_and_kpis(n_clicks, org_val, region_val, fac_vals, start_dt, end_dt):
    empty_fig = {'layout': {'title': {'x':0.5}}} # Basic empty fig with centered title
    if n_clicks == 0: return [], {**empty_fig, 'layout': {**empty_fig['layout'], 'title': {'text': 'Daily Trends'}}}, \
                                 {**empty_fig, 'layout': {**empty_fig['layout'], 'title': {'text': 'Facility Comparison'}}}, \
                                 {**empty_fig, 'layout': {**empty_fig['layout'], 'title': {'text': 'Org/Region Summary'}}}

    kpi_children, daily_fig, fac_fig, org_fig = [], {**empty_fig, 'layout': {**empty_fig['layout'], 'title': {'text': 'Daily Trends: No Data'}}}, \
                                                     {**empty_fig, 'layout': {**empty_fig['layout'], 'title': {'text': 'Facility Comparison: No Data'}}}, \
                                                     {**empty_fig, 'layout': {**empty_fig['layout'], 'title': {'text': 'Org/Region Summary: No Data'}}}

    df_daily = pd.DataFrame()
    # Fetch daily summary data if facilities are selected or if no org/region is selected (implying all facilities for date range)
    if fac_vals or (not org_val and not region_val):
        temp_daily_dfs = []
        if fac_vals: # Specific facilities
            for fac_id in fac_vals:
                temp_daily_dfs.append(metrics_queries.get_daily_processing_summary_metrics(facility_id=fac_id, start_date=start_dt, end_date=end_dt))
        else: # All facilities for the date range
            temp_daily_dfs.append(metrics_queries.get_daily_processing_summary_metrics(start_date=start_dt, end_date=end_dt))
        if temp_daily_dfs: df_daily = pd.concat(temp_daily_dfs).groupby('summary_date', as_index=False).sum()

    df_org = metrics_queries.get_org_level_processing_metrics(org_id=org_val, region_id=region_val, start_date=start_dt, end_date=end_dt)

    # KPIs
    source_df_kpi = df_daily if not df_daily.empty else df_org # Prioritize daily summary if available for KPIs
    if not source_df_kpi.empty:
        total_proc = source_df_kpi['total_claims_processed'].sum()
        total_fail = source_df_kpi['total_claims_failed'].sum()
        total_charge = source_df_kpi.get('total_charge_amount', pd.Series(0)).sum() # Handle missing columns
        total_reimb = source_df_kpi.get('total_reimbursement_amount', pd.Series(0)).sum()
        err_rate = (total_fail / total_proc * 100) if total_proc else 0
        avg_charge = (total_charge / total_proc) if total_proc else 0
        avg_reimb = (total_reimb / total_proc) if total_proc else 0
        kpi_children = [
            html.Div([html.H5("Total Processed", className="kpi-title"), html.P(f"{total_proc:,.0f}", className="kpi-value")]),
            html.Div([html.H5("Total Failed", className="kpi-title"), html.P(f"{total_fail:,.0f}", className="kpi-value")]),
            html.Div([html.H5("Error Rate", className="kpi-title"), html.P(f"{err_rate:.2f}%", className="kpi-value")]),
            html.Div([html.H5("Avg Charge/Claim", className="kpi-title"), html.P(f"${avg_charge:,.2f}", className="kpi-value")]),
            html.Div([html.H5("Avg Reimb./Claim", className="kpi-title"), html.P(f"${avg_reimb:,.2f}", className="kpi-value")])
        ]
    else: kpi_children = [html.P("No KPI data for current filters.")]

    # Daily Trends
    if not df_daily.empty and 'summary_date' in df_daily.columns:
        df_daily['reimbursement_rate'] = (df_daily['total_reimbursement_amount'] / df_daily['total_charge_amount']).fillna(0) * 100
        df_daily.loc[df_daily['total_charge_amount'] == 0, 'reimbursement_rate'] = 0

        daily_fig = make_subplots(specs=[[{"secondary_y": True}]])
        daily_fig.add_trace(px.line(df_daily, x='summary_date', y='total_claims_processed').data[0].update(name='Processed'), secondary_y=False)
        daily_fig.add_trace(px.line(df_daily, x='summary_date', y='total_claims_failed').data[0].update(name='Failed', line_color='red'), secondary_y=False)
        daily_fig.add_trace(px.line(df_daily, x='summary_date', y='reimbursement_rate').data[0].update(name='Reimb. Rate (%)', line_dash='dash'), secondary_y=True)
        daily_fig.update_layout(title_text="Daily Processing Trends & Reimbursement Rate", title_x=0.5)
        daily_fig.update_yaxes(title_text="Claims Count", secondary_y=False); daily_fig.update_yaxes(title_text="Reimb. Rate (%)", secondary_y=True, range=[0,100])

    # Facility Comparison (using df_daily which is already filtered by date and potentially facilities)
    if not df_daily.empty and fac_vals and 'facility_id' in df_daily.columns: # Only show if specific facilities were selected
        # If df_daily was from multiple facilities, group again
        df_fac_comp = df_daily.groupby('facility_id').agg(
            total_claims_processed=('total_claims_processed', 'sum'),
            error_rate_percentage=('error_rate_percentage', 'mean')
        ).reset_index()
        # TODO: map facility_id to name
        fac_fig = px.bar(df_fac_comp, x='facility_id', y='total_claims_processed', title="Facility Comparison: Processed Claims", labels={'facility_id':'Facility'})
        fac_fig.update_layout(title_x=0.5)

    # Org/Region Summary
    if not df_org.empty:
        x_ax = 'org_name' if not region_val else 'region_name'
        if x_ax not in df_org.columns: x_ax = 'org_name' # Fallback

        df_org_melt = df_org.melt(id_vars=[x_ax], value_vars=['total_claims_processed', 'total_claims_failed'], var_name='metric', value_name='count')
        org_fig = px.bar(df_org_melt, x=x_ax, y='count', color='metric', barmode='group', title="Organizational/Regional Summary")
        org_fig.update_layout(title_x=0.5)

    return kpi_children, daily_fig, fac_fig, org_fig

# --- Callbacks for Healthcare Analytics Tab ---
@app.callback(Output('facility-filter-ha', 'options'), [Input('main-tabs', 'value')])
def update_ha_filter_options(main_tab_value):
    if main_tab_value == 'tab-healthcare-analytics':
        try: return claims_queries.get_distinct_facilities()
        except Exception as e: print(f"Error HA facility dropdown: {e}"); return []
    return []

@app.callback(
    Output('healthcare-analytics-content', 'children'),
    [Input('apply-filters-ha-button', 'n_clicks'), Input('healthcare-analytics-subtabs', 'value')],
    [State('facility-filter-ha', 'value'), State('date-picker-range-ha', 'start_date'),
     State('date-picker-range-ha', 'end_date'), State('top-n-filter-ha', 'value')]
)
def render_healthcare_analytics_sub_content(n_clicks, subtab_val, fac_vals, start_dt, end_dt, top_n_val):
    if n_clicks == 0 : return html.P("Select a sub-tab and apply filters.")
    top_n = int(top_n_val) if top_n_val and str(top_n_val).isdigit() and int(top_n_val) > 0 else 10

    dt_props = {'page_size': 10, 'sort_action': 'native', 'filter_action': 'native', 'page_action': 'native',
                'export_format': 'csv', 'fill_width': False,
                'style_cell': {'textAlign': 'left', 'padding': '5px', 'fontFamily': 'Arial, sans-serif'},
                'style_header': {'backgroundColor': '#ecf0f1', 'fontWeight': 'bold', 'borderBottom': '1px solid black'},
                'style_data': {'whiteSpace': 'normal', 'height': 'auto', 'borderBottom': '1px solid #eee'},
                'style_table': {'overflowX': 'auto'}}

    def empty_layout(title): return html.Div(className='graph-container', children=[html.H5(title, style={'textAlign':'center'}), html.P("No data for filters.", style={'textAlign':'center'})])

    if subtab_val == 'subtab-cpt':
        df = metrics_queries.get_cpt_code_analytics(facility_id=fac_vals, start_date=start_dt, end_date=end_dt, top_n=top_n)
        if df.empty: return empty_layout(f"Top {top_n} CPT Code Analysis")
        tbl = dash_table.DataTable(id='cpt-tbl', columns=[{'name':c.replace('_',' ').title(), 'id':c} for c in df.columns], data=df.to_dict('records'), **dt_props)
        fig = px.bar(df, x='procedure_code', y='cpt_volume', title=f"Top {top_n} CPT Codes by Volume")
        fig.update_layout(title_x=0.5)
        return [html.H5(f"Top {top_n} CPT Codes", style={'textAlign':'center'}), html.Div(className='table-container', children=[tbl]), html.Div(className='graph-container', children=[dcc.Graph(figure=fig)])]

    elif subtab_val == 'subtab-dx':
        df = metrics_queries.get_diagnosis_code_analytics(facility_id=fac_vals, start_date=start_dt, end_date=end_dt, top_n=top_n)
        if df.empty: return empty_layout(f"Top {top_n} Diagnosis Code Analysis")
        tbl = dash_table.DataTable(id='dx-tbl', columns=[{'name':c.replace('_',' ').title(), 'id':c} for c in df.columns], data=df.to_dict('records'), **dt_props)
        fig = px.bar(df, x='diagnosis_code', y='diagnosis_volume', title=f"Top {top_n} Diagnosis Codes by Volume", hover_data=['diagnosis_description'])
        fig.update_layout(title_x=0.5)
        return [html.H5(f"Top {top_n} Diagnosis Codes", style={'textAlign':'center'}), html.Div(className='table-container', children=[tbl]), html.Div(className='graph-container', children=[dcc.Graph(figure=fig)])]

    elif subtab_val == 'subtab-payer':
        df = metrics_queries.get_payer_analytics(facility_id=fac_vals, start_date=start_dt, end_date=end_dt)
        if df.empty: return empty_layout("Payer Analysis")
        if 'distinct_claims_for_payer' in df.columns and df['distinct_claims_for_payer'].sum() > 0:
            df['avg_charge_per_claim'] = (df['total_charges_for_payer'] / df['distinct_claims_for_payer']).fillna(0)
        else: df['avg_charge_per_claim'] = 0
        tbl = dash_table.DataTable(id='payer-tbl', columns=[{'name':c.replace('_',' ').title(), 'id':c} for c in df.columns], data=df.to_dict('records'), **dt_props)
        fig = px.pie(df, names='payer_name', values='distinct_claims_for_payer', title="Claim Volume by Payer", hover_data=['total_charges_for_payer', 'avg_charge_per_claim'])
        fig.update_traces(textposition='inside', textinfo='percent+label'); fig.update_layout(title_x=0.5)
        return [html.H5("Payer Analysis", style={'textAlign':'center'}), html.Div(className='table-container', children=[tbl]), html.Div(className='graph-container', children=[dcc.Graph(figure=fig)])]

    elif subtab_val == 'subtab-provider':
        df = metrics_queries.get_provider_summary_analytics(facility_id=fac_vals, start_date=start_dt, end_date=end_dt, top_n=top_n)
        if df.empty: return empty_layout(f"Top {top_n} Provider Metrics")
        if 'provider_first_name' in df.columns and 'provider_last_name' in df.columns:
            df['provider_full_name'] = df['provider_first_name'] + ' ' + df['provider_last_name']
            cols = list(df.columns); cols.insert(cols.index('provider_first_name'), cols.pop(cols.index('provider_full_name'))); df=df[cols]
        tbl_cols = [{'name':c.replace('_',' ').title(),'id':c} for c in df.columns if c not in ['provider_first_name','provider_last_name','rendering_provider_id'] or c == 'provider_full_name']
        tbl = dash_table.DataTable(id='provider-tbl', columns=tbl_cols, data=df.to_dict('records'), page_size=top_n, **dt_props)
        fig = px.bar(df, x='provider_full_name', y='total_provider_charges', title=f"Top {top_n} Providers by Total Charges")
        fig.update_xaxes(categoryorder="total descending"); fig.update_layout(title_x=0.5)
        return [html.H5(f"Top {top_n} Provider Metrics", style={'textAlign':'center'}), html.Div(className='table-container', children=[tbl]), html.Div(className='graph-container', children=[dcc.Graph(figure=fig)])]

    elif subtab_val == 'subtab-demographics':
        df = metrics_queries.get_patient_demographics(facility_id=fac_vals, start_date=start_dt, end_date=end_dt)
        if df.empty: return empty_layout("Patient Demographics")

        fig_age, fig_gender = {'layout': {'title': {'text':'Age Distribution: No Data', 'x':0.5}}}, {'layout': {'title': {'text':'Gender Distribution: No Data', 'x':0.5}}}
        if 'age' in df.columns and not df['age'].dropna().empty:
            bins = list(range(0,101,10)); labels=[f'{i}-{i+9}' for i in bins[:-1]]
            df['age_group'] = pd.cut(df['age'].dropna(), bins=bins, labels=labels, right=False)
            if not df['age_group'].dropna().empty:
                age_counts = df['age_group'].value_counts().sort_index().reset_index(); age_counts.columns=['age_group','count']
                fig_age = px.bar(age_counts, x='age_group', y='count', title="Patient Age Distribution")
                fig_age.update_layout(title_x=0.5)
        if 'gender' in df.columns and not df['gender'].dropna().empty:
            gender_counts = df['gender'].value_counts().reset_index(); gender_counts.columns=['gender','count']
            fig_gender = px.pie(gender_counts, names='gender', values='count', title="Patient Gender Distribution")
            fig_gender.update_traces(textposition='inside', textinfo='percent+label'); fig_gender.update_layout(title_x=0.5)
        return [html.H5("Patient Demographics", style={'textAlign':'center'}), html.Div(className='graph-container', children=[dcc.Graph(figure=fig_age)]), html.Div(className='graph-container', children=[dcc.Graph(figure=fig_gender)])]

    return html.P("Select a sub-tab and apply filters.")

print("All callbacks defined, including styling and new Provider Metrics tab logic.")
