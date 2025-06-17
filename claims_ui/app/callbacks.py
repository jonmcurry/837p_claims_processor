from dash import dcc, html, dash_table # Ensure dash_table is imported from dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate # For preventing updates when not needed
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- App Instance ---
from claims_ui.app import app 

# --- Layout Imports ---
from .layouts.processed_claims_layout import create_processed_claims_layout
from .layouts.failed_claims_layout import create_failed_claims_layout
from .layouts.processing_metrics_layout import create_processing_metrics_layout
from .layouts.healthcare_analytics_layout import create_healthcare_analytics_layout

# --- Data Query Imports ---
from .queries import claims_queries, metrics_queries

# --- Constants for DataTable styling and properties ---
COMMON_DATATABLE_PROPS = {
    'page_size': 15,
    'sort_action': 'native',
    'filter_action': 'native',
    'row_selectable': 'single',
    'style_cell': {'textAlign': 'left', 'padding': '10px', 'fontFamily': 'var(--primary-font)'}, # Use CSS var
    'style_header': {
        'backgroundColor': '#f8f9fc', 
        'fontWeight': '600', 
        'borderBottom': '2px solid var(--border-color)', # Use CSS var
        'padding': '12px 10px',
        'color': 'var(--text-color-primary)'
    },
    'style_data': {
        'whiteSpace': 'normal', 
        'height': 'auto', 
        'borderBottom': '1px solid #f1f3f7', # Use CSS var
        'color': 'var(--text-color-secondary)'
    },
    'export_format': 'csv',
    'fill_width': False, 
}

# --- Helper Function for Empty Charts ---
def create_empty_figure(message="No data to display for the current selection."):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": message, "xref": "paper", "yref": "paper",
            "showarrow": False, "font": {"size": 14, "color": "var(--text-color-secondary)"} 
        }],
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# =========================================================================
# Main Page Navigation and Content Display
# =========================================================================
@app.callback(
    [Output('page-content', 'children'), 
     Output('content-title', 'children')], 
    [Input('url', 'pathname')]
)
def display_page_content(pathname):
    if pathname == '/processed':
        return create_processed_claims_layout(), "Processed Claims Dashboard"
    elif pathname == '/failed':
        return create_failed_claims_layout(), "Failed Claims Analysis"
    elif pathname == '/analytics':
        return create_healthcare_analytics_layout(), "Healthcare Analytics Hub"
    return create_processing_metrics_layout(), "Processing Metrics Overview"

@app.callback(
    [Output(f"nav-link-{page_key}", "className") for page_key in ["home", "analytics", "processed", "failed"]],
    [Input("url", "pathname")],
    prevent_initial_call=True 
)
def update_active_sidebar_links(pathname):
    links_map = {"/": "home", "/analytics": "analytics", "/processed": "processed", "/failed": "failed"}
    active_page_key = links_map.get(pathname)
    class_names = []
    for page_key in ["home", "analytics", "processed", "failed"]:
        class_names.append("nav-link active" if page_key == active_page_key else "nav-link")
    return class_names

# =========================================================================
# Callbacks for Dropdown Population
# =========================================================================
@app.callback(Output('facility-filter-pm', 'options'), [Input('url', 'pathname')], prevent_initial_call=True)
def populate_pm_page_filters(pathname):
    if pathname == '/': return claims_queries.get_distinct_facilities()
    raise PreventUpdate

@app.callback([Output('facility-filter-pc', 'options'), Output('payer-filter-pc', 'options')], [Input('url', 'pathname')], prevent_initial_call=True)
def populate_pc_page_filters(pathname):
    if pathname == '/processed': return claims_queries.get_distinct_facilities(), claims_queries.get_distinct_payers()
    raise PreventUpdate

@app.callback([Output('facility-filter-fc', 'options'), Output('failure-category-filter-fc', 'options')], [Input('url', 'pathname')], prevent_initial_call=True)
def populate_fc_page_filters(pathname):
    if pathname == '/failed': return claims_queries.get_distinct_facilities(), claims_queries.get_distinct_failure_categories()
    raise PreventUpdate

@app.callback(Output('facility-filter-ha', 'options'), [Input('url', 'pathname')], prevent_initial_call=True)
def populate_ha_page_filters(pathname):
    if pathname == '/analytics': return claims_queries.get_distinct_facilities()
    raise PreventUpdate

# =========================================================================
# Callbacks for Processing Metrics Page ('/')
# =========================================================================
@app.callback(
    [Output('kpi-summary-pm', 'children'),
     Output('daily-trends-graph-pm', 'figure'),
     Output('facility-comparison-graph-pm', 'figure'),
     Output('org-region-summary-graph-pm', 'figure')],
    [Input('apply-filters-pm-button', 'n_clicks')],
    [State('org-filter-pm', 'value'), State('region-filter-pm', 'value'),
     State('facility-filter-pm', 'value'), State('date-picker-range-pm', 'start_date'),
     State('date-picker-range-pm', 'end_date'), State('url', 'pathname')]
)
def update_processing_metrics_page_content(n_clicks, org_id, region_id, facility_values, start_date, end_date, pathname):
    if pathname != '/': raise PreventUpdate

    if start_date is None and end_date is None:
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    df_daily_summary = metrics_queries.get_daily_processing_summary_metrics(facility_id=facility_values, start_date=start_date, end_date=end_date)
    df_org_summary = metrics_queries.get_org_level_processing_metrics(org_id=org_id, region_id=region_id, start_date=start_date, end_date=end_date)

    kpi_children = []
    summary_for_kpis = df_org_summary if org_id or region_id else df_daily_summary
    if not summary_for_kpis.empty:
        total_processed = int(summary_for_kpis['total_claims_processed'].sum())
        total_failed = int(summary_for_kpis['total_claims_failed'].sum())
        total_charges = summary_for_kpis['total_charge_amount'].sum()
        total_reimbursement = summary_for_kpis['total_reimbursement_amount'].sum()
        error_rate = (total_failed / total_processed * 100) if total_processed else 0
        avg_charge = total_charges / total_processed if total_processed else 0
        avg_reimb = total_reimbursement / total_processed if total_processed else 0
        kpis_data = {
            "Total Processed": f"{total_processed:,}", "Total Failed": f"{total_failed:,}",
            "Error Rate": f"{error_rate:.2f}%", "Total Billed": f"${total_charges:,.2f}",
            "Avg Billed/Claim": f"${avg_charge:,.2f}", "Avg Reimb./Claim": f"${avg_reimb:,.2f}",
        }
        for title, value in kpis_data.items():
            kpi_children.append(html.Div([html.Span(title, className='kpi-title'), html.Span(value, className='kpi-value')], className='kpi-card dashboard-card'))
    else:
        kpi_children = [html.Div(html.P("No KPI data for current filters."), className="dashboard-card kpi-card")]

    fig_daily = create_empty_figure()
    if not df_daily_summary.empty and 'summary_date' in df_daily_summary.columns:
        daily_agg = df_daily_summary.groupby('summary_date').agg(total_claims_processed=('total_claims_processed', 'sum'), total_claims_failed=('total_claims_failed', 'sum')).reset_index()
        if not daily_agg.empty:
            fig_daily = px.area(daily_agg, x='summary_date', y=['total_claims_processed', 'total_claims_failed'], title="Daily Claim Volume Trends", template="plotly_white", labels={'summary_date': 'Date', 'value': 'Claims Count', 'variable': 'Status'})
            fig_daily.update_layout(title_x=0.5, legend_title_text='')
    
    fig_facility = create_empty_figure()
    if not df_daily_summary.empty and facility_values and 'facility_id' in df_daily_summary.columns:
        facility_plot_df = df_daily_summary[df_daily_summary['facility_id'].isin(facility_values)]
        if not facility_plot_df.empty:
            facility_agg = facility_plot_df.groupby('facility_id').agg(total_claims_processed=('total_claims_processed', 'sum')).reset_index()
            if not facility_agg.empty:
                fig_facility = px.bar(facility_agg, x='facility_id', y='total_claims_processed', title="Facility Claim Volume Comparison", template="plotly_white", color='facility_id', labels={'facility_id': 'Facility', 'total_claims_processed': 'Total Claims'})
                fig_facility.update_layout(title_x=0.5, showlegend=False)

    fig_org_region = create_empty_figure()
    if not df_org_summary.empty:
        plot_df, x_col, title_prefix = df_org_summary, 'org_name', 'Organization'
        if org_id and region_id:
            plot_df = df_org_summary[df_org_summary['region_id'] == region_id]
            x_col, title_prefix = 'region_name', plot_df[x_col].iloc[0] if not plot_df.empty else "Region"
        elif org_id:
            plot_df = df_org_summary[df_org_summary['org_id'] == org_id]
            x_col = 'region_name' if 'region_name' in plot_df.columns and plot_df['region_name'].nunique() > 1 else 'org_name'
            if x_col == 'org_name': plot_df = plot_df.groupby('org_name').sum().reset_index()
            title_prefix = plot_df['org_name'].iloc[0] if not plot_df.empty else "Organization"
        if not plot_df.empty:
            fig_org_region = px.bar(plot_df, x=x_col, y=['total_claims_processed', 'total_claims_failed'], title=f"{title_prefix} Processing Summary", template="plotly_white", barmode='group', labels={x_col: x_col.replace('_',' ').title(), 'value':'Claims Count'})
            fig_org_region.update_layout(title_x=0.5, legend_title_text='Metric')
            
    return kpi_children, fig_daily, fig_facility, fig_org_region

# =========================================================================
# Callbacks for Processed Claims Page ('/processed')
# =========================================================================
@app.callback(
    [Output('processed-claims-table', 'data'), Output('processed-claims-table', 'columns')],
    [Input('apply-filters-pc-button', 'n_clicks')],
    [State('facility-filter-pc', 'value'), State('payer-filter-pc', 'value'),
     State('date-picker-range-pc', 'start_date'), State('date-picker-range-pc', 'end_date'),
     State('url', 'pathname')]
)
def update_processed_claims_table_content(n_clicks, facility_values, payer_values, start_date, end_date, pathname):
    if pathname != '/processed': raise PreventUpdate
    df = claims_queries.get_processed_claims(facility_id=facility_values, payer_id=payer_values, start_date=start_date, end_date=end_date, limit=250)
    if df.empty: return [], [] 
    return df.to_dict('records'), [{'name': col.replace('_', ' ').title(), 'id': col} for col in df.columns]

@app.callback(Output('selected-claim-details-pc', 'children'), [Input('processed-claims-table', 'active_cell')], [State('processed-claims-table', 'data')])
def display_processed_claim_details(active_cell, table_data):
    if not active_cell or not table_data: return html.Div([html.P("Select a claim to view details.", style={'padding': '20px'})])
    selected_row_idx = active_cell['row']
    if selected_row_idx < len(table_data):
        claim = table_data[selected_row_idx]
        details_content = [html.P(f"{key.replace('_', ' ').title()}: {value}") for key, value in claim.items()]
        return [html.Div(html.H4("Selected Claim Details"), className="dashboard-card-header"), html.Div(details_content, className="dashboard-card-content")]
    return html.P("Could not retrieve claim details.")

# =========================================================================
# Callbacks for Failed Claims Page ('/failed')
# =========================================================================
@app.callback(
    [Output('failed-claims-table', 'data'), Output('failed-claims-table', 'columns'), Output('failure-reason-bargraph-fc', 'figure')],
    [Input('apply-filters-fc-button', 'n_clicks')],
    [State('facility-filter-fc', 'value'), State('failure-category-filter-fc', 'value'),
     State('date-picker-range-fc', 'start_date'), State('date-picker-range-fc', 'end_date'), 
     State('url', 'pathname')]
)
def update_failed_claims_page_content(n_clicks, facility_values, category_values, start_date, end_date, pathname):
    if pathname != '/failed': raise PreventUpdate
    df = claims_queries.get_failed_claims(facility_id=facility_values, failure_category=category_values, start_date=start_date, end_date=end_date, limit=250)
    if df.empty: return [], [], create_empty_figure("No failed claims match filters.")
    table_cols = [{'name': col.replace('_', ' ').title(), 'id': col} for col in df.columns]
    fig_failure = create_empty_figure("No data for failure category graph.")
    if 'failure_category' in df.columns and not df['failure_category'].dropna().empty:
        counts = df['failure_category'].value_counts().reset_index(); counts.columns = ['category', 'count']
        fig_failure = px.bar(counts, x='category', y='count', title="Failed Claims by Category", template="plotly_white", color='category', labels={'category':'Failure Category', 'count':'Number of Claims'})
        fig_failure.update_layout(title_x=0.5, showlegend=False)
    return df.to_dict('records'), table_cols, fig_failure

@app.callback(Output('selected-failed-claim-details-fc', 'children'),[Input('failed-claims-table', 'active_cell')],[State('failed-claims-table', 'data')])
def display_failed_claim_details(active_cell, table_data):
    if not active_cell or not table_data: return html.Div([html.P("Select a failed claim to view details.", style={'padding': '20px'})])
    selected_row_idx = active_cell['row']
    if selected_row_idx < len(table_data):
        claim = table_data[selected_row_idx]
        details_content = [html.P(f"{key.replace('_', ' ').title()}: {value}") for key, value in claim.items()]
        return [html.Div(html.H4("Selected Failed Claim Details"), className="dashboard-card-header"), html.Div(details_content, className="dashboard-card-content")]
    return html.P("Could not retrieve failed claim details.")

# =========================================================================
# Callbacks for Healthcare Analytics Page ('/analytics')
# =========================================================================
@app.callback(
    Output('healthcare-analytics-content', 'children'),
    [Input('apply-filters-ha-button', 'n_clicks'), Input('healthcare-analytics-subtabs', 'value')],
    [State('facility-filter-ha', 'value'), State('date-picker-range-ha', 'start_date'),
     State('date-picker-range-ha', 'end_date'), State('top-n-filter-ha', 'value'),
     State('url', 'pathname')]
)
def render_healthcare_analytics_subpage_content(n_clicks, subtab_value, facility_values, start_date, end_date, top_n, pathname):
    if pathname != '/analytics' or n_clicks is None or n_clicks == 0: 
        return html.Div(html.P("Select filters and click 'Apply Filters' to view analytics."), className="dashboard-card dashboard-card-content text-center")
    try: top_n = int(top_n) if top_n and int(top_n) > 0 else 10
    except ValueError: top_n = 10
    
    card_content_list = []
    empty_card = lambda msg: html.Div([html.Div(html.H4(msg.split(" ")[0] + " Data"), className="dashboard-card-header"), html.Div(html.P(f"No {msg.lower()} for current filters."),className="dashboard-card-content")], className="dashboard-card")

    if subtab_value == 'subtab-cpt':
        df = metrics_queries.get_cpt_code_analytics(facility_values, start_date, end_date, top_n)
        if not df.empty:
            table = dash_table.DataTable(id='cpt-table', data=df.to_dict('records'), columns=[{'name':c.replace('_',' ').title(),'id':c} for c in df.columns], **COMMON_DATATABLE_PROPS)
            fig = px.bar(df,x='procedure_code',y='cpt_volume',title=f"Top {top_n} CPT Codes by Volume",template="plotly_white",color='procedure_code',labels={'procedure_code':'CPT Code','cpt_volume':'Volume'}).update_layout(title_x=0.5,showlegend=False)
            card_content_list.append(html.Div([html.Div(html.H4("CPT Code Data"),className="dashboard-card-header"),html.Div(table,className="dashboard-card-content")], className="dashboard-card"))
            card_content_list.append(html.Div([html.Div(html.H4(f"Top {top_n} CPT Graph"),className="dashboard-card-header"),html.Div(dcc.Graph(figure=fig),className="dashboard-card-content")], className="dashboard-card"))
        else: card_content_list.append(empty_card("CPT Data"))
    
    elif subtab_value == 'subtab-dx':
        df = metrics_queries.get_diagnosis_code_analytics(facility_values, start_date, end_date, top_n)
        if not df.empty:
            table = dash_table.DataTable(id='dx-table', data=df.to_dict('records'), columns=[{'name':c.replace('_',' ').title(),'id':c} for c in df.columns], **COMMON_DATATABLE_PROPS)
            fig = px.bar(df,x='diagnosis_code',y='diagnosis_volume',title=f"Top {top_n} Dx Codes by Volume",hover_data=['diagnosis_description'],template="plotly_white",color='diagnosis_code',labels={'diagnosis_code':'Dx Code','diagnosis_volume':'Volume'}).update_layout(title_x=0.5,showlegend=False)
            card_content_list.append(html.Div([html.Div(html.H4("Diagnosis Code Data"),className="dashboard-card-header"),html.Div(table,className="dashboard-card-content")], className="dashboard-card"))
            card_content_list.append(html.Div([html.Div(html.H4(f"Top {top_n} Dx Graph"),className="dashboard-card-header"),html.Div(dcc.Graph(figure=fig),className="dashboard-card-content")], className="dashboard-card"))
        else: card_content_list.append(empty_card("Diagnosis Data"))

    elif subtab_value == 'subtab-payer':
        df = metrics_queries.get_payer_analytics(facility_values, start_date, end_date)
        if not df.empty:
            df['avg_charge_per_claim'] = ((df['total_charges_for_payer']/df['distinct_claims_for_payer']).fillna(0)).round(2)
            cols = [{'name':'Payer','id':'payer_name'},{'name':'Claims','id':'distinct_claims_for_payer'},{'name':'Total Billed','id':'total_charges_for_payer','type':'numeric','format':dash_table.Format.Format(prefix='$',precision=2)},{'name':'Avg Billed/Claim','id':'avg_charge_per_claim','type':'numeric','format':dash_table.Format.Format(prefix='$',precision=2)}]
            table = dash_table.DataTable(id='payer-table', data=df.to_dict('records'), columns=cols, **COMMON_DATATABLE_PROPS)
            fig = px.pie(df,names='payer_name',values='distinct_claims_for_payer',title="Claim Volume by Payer",hole=0.4,template="plotly_white").update_layout(title_x=0.5)
            card_content_list.append(html.Div([html.Div(html.H4("Payer Data"),className="dashboard-card-header"),html.Div(table,className="dashboard-card-content")], className="dashboard-card"))
            card_content_list.append(html.Div([html.Div(html.H4("Payer Distribution - Claims Volume"),className="dashboard-card-header"),html.Div(dcc.Graph(figure=fig),className="dashboard-card-content")], className="dashboard-card"))
        else: card_content_list.append(empty_card("Payer Data"))

    elif subtab_value == 'subtab-demographics':
        df = metrics_queries.get_patient_demographics(facility_values, start_date, end_date)
        if not df.empty:
            if 'age' in df.columns and not df['age'].dropna().empty:
                bins, labels = [0,10,18,30,40,50,60,70,80,120], ['0-10','11-18','19-30','31-40','41-50','51-60','61-70','71-80','80+']
                df['age_group'] = pd.cut(df['age'].dropna().astype(int), bins=bins, labels=labels, right=False)
                age_counts = df['age_group'].value_counts().sort_index().reset_index(); age_counts.columns = ['age_group', 'count']
                fig_age = px.bar(age_counts,x='age_group',y='count',title="Patient Age Groups",template="plotly_white").update_layout(title_x=0.5)
                card_content_list.append(html.Div([html.Div(html.H4("Patient Age Distribution"),className="dashboard-card-header"),html.Div(dcc.Graph(figure=fig_age),className="dashboard-card-content")], className="dashboard-card"))
            else: card_content_list.append(empty_card("Patient Age Data"))
            if 'gender' in df.columns and not df['gender'].dropna().empty:
                gender_counts = df['gender'].value_counts().reset_index(); gender_counts.columns = ['gender', 'count']
                fig_gender = px.pie(gender_counts,names='gender',values='count',title="Patient Gender Distribution",hole=0.4,template="plotly_white").update_layout(title_x=0.5)
                card_content_list.append(html.Div([html.Div(html.H4("Patient Gender Distribution"),className="dashboard-card-header"),html.Div(dcc.Graph(figure=fig_gender),className="dashboard-card-content")], className="dashboard-card"))
            else: card_content_list.append(empty_card("Patient Gender Data"))
        else: card_content_list.append(empty_card("Demographics Data"))

    elif subtab_value == 'subtab-provider':
        df = metrics_queries.get_provider_summary_analytics(facility_values, start_date, end_date, top_n)
        if not df.empty:
            df['provider_full_name'] = df['provider_first_name'].fillna('') + ' ' + df['provider_last_name'].fillna('')
            cols = [{'name':'Provider','id':'provider_full_name'},{'name':'Patients','id':'unique_patients_count'},{'name':'Units','id':'total_units'},{'name':'Billed','id':'total_provider_charges','type':'numeric','format':dash_table.Format.Format(prefix='$',precision=2)},{'name':'RVUs','id':'total_provider_rvus','type':'numeric','format':dash_table.Format.Format(precision=2)}]
            table = dash_table.DataTable(id='provider-table', data=df.to_dict('records'), columns=cols, **COMMON_DATATABLE_PROPS)
            fig = px.bar(df,x='provider_full_name',y='total_provider_charges',title=f"Top {top_n} Providers by Billed Amount",template="plotly_white",color='provider_full_name',labels={'provider_full_name':'Provider','total_provider_charges':'Total Billed'}).update_layout(title_x=0.5,showlegend=False)
            card_content_list.append(html.Div([html.Div(html.H4("Provider Metrics Data"),className="dashboard-card-header"),html.Div(table,className="dashboard-card-content")], className="dashboard-card"))
            card_content_list.append(html.Div([html.Div(html.H4(f"Top {top_n} Providers by Billed Graph"),className="dashboard-card-header"),html.Div(dcc.Graph(figure=fig),className="dashboard-card-content")], className="dashboard-card"))
        else: card_content_list.append(empty_card("Provider Data"))

    return html.Div(card_content_list) if card_content_list else html.Div(html.P("Select a category and apply filters to view analytics."), className="dashboard-card dashboard-card-content text-center")

# print("claims_ui.app.callbacks loaded")