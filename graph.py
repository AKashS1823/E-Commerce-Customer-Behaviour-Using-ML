import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Load the dataset
df = pd.read_csv("ecommerce_customer_data_large.csv")

# Data overview
print(df.shape)
print(df.info())
print(df.head())
print(df.nunique())
print(df[df.duplicated(keep=False)])

# Data Cleaning and Transformation
df['Returns'] = df['Returns'].fillna(0).astype(int)
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df['Purchase Year'] = df['Purchase Date'].dt.year
df['Purchase Month'] = df['Purchase Date'].dt.month_name()
df['Total Price'] = df['Product Price'] * df['Quantity']

# Select relevant columns
cols = ['Customer ID', 'Customer Name', 'Purchase Date', 'Purchase Year', 'Purchase Month', 'Product Category', 'Product Price', 'Quantity', 'Total Price', 'Payment Method', 'Returns', 'Gender', 'Customer Age', 'Churn']
df = df[cols]

# Save cleaned data
df.to_csv('ecommerce_data_cleaned.csv', index=False)

# Summary Statistics
df.describe(include='all')
df.describe(include=['object'])

# Grouping and Aggregation
gender_summary = df.groupby('Gender').agg(
    total_count=('Customer ID', 'size'),
    min_amount=('Total Price', 'min'),
    max_amount=('Total Price', 'max'),
    median_amount=('Total Price', 'median'),
    total_amount=('Total Price', 'sum'),
    average_amount=('Total Price', 'mean'),
    total_returned=('Returns', 'sum'),
    return_rates=('Returns', 'mean'),
    total_churned=('Churn', 'sum'),
    churn_rates=('Churn', 'mean')
).reset_index()

# Calculate total_amount_percentage
sum_amount = gender_summary['total_amount'].sum()
gender_summary['total_amount_percentage'] = (gender_summary['total_amount'] / sum_amount * 100).round(2)

# Visualizations
fig1 = px.pie(values=df['Gender'].value_counts(), names=df['Gender'].value_counts().index,
              color=df['Gender'].value_counts().index,
              color_discrete_map={'Female': '#FF6692', 'Male': '#3366CC'},
              title='1. Gender Distribution')
fig1.update_traces(textposition='inside', textinfo='text', texttemplate='%{label}<br>%{value}<br>(%{percent})')
fig1.update_layout(title={'x': 0.5, 'y': 0.9}, width=400, height=400, showlegend=False)
pio.write_html(fig1, file='templates/gender_distribution.html', auto_open=False)

fig2 = px.bar(gender_summary, x='total_amount', y='Gender',
              orientation='h',
              color='Gender',
              color_discrete_map={'Female': '#FF6692', 'Male': '#3366CC'},
              title='2. Total Purchase Amount by Gender',
              text=gender_summary.apply(lambda x: f"${x['total_amount']:,}<br>({x['total_amount_percentage']:.2f}%)", axis=1),
              labels={'total_amount': 'Total Purchased Amount ($)'})
fig2.update_layout(title={'x': 0.5, 'y': 0.9}, yaxis_title=None, width=550, height=350, showlegend=False)
pio.write_html(fig2, file='templates/purchase_amount_gender.html', auto_open=False)

overall_avg_price = df['Total Price'].mean()
avg_price_gender_year = df.groupby(['Gender', 'Purchase Year'], as_index=False)['Total Price'].mean()

fig3 = px.box(df, x='Gender', y='Total Price', color='Purchase Year',
              title='3. Total Price Spent by Gender, Breakdown by Year',
              labels={'Total Price': 'Total Purchased Amount ($)'})
fig3.add_hline(y=overall_avg_price, line_dash="dash", line_color="#325A9B",
               annotation_text=f'Overall Average: ${overall_avg_price:.2f}',
               annotation_position="top", annotation_font_color="#325A9B")
fig3.update_layout(title={'x': 0.5, 'y': 0.9}, width=700, height=450, xaxis_title=None)
pio.write_html(fig3, file='templates/price_spent_gender_year.html', auto_open=False)

gender_summary['churn_rates'] *= 100
gender_summary['return_rates'] *= 100

fig4 = px.bar(gender_summary, x='Gender', y=['total_churned', 'total_returned'],
              barmode='group',
              title='4. Customer Churn and Returns by Gender',
              text_auto='.4s',
              labels={'value': 'Number of Customers'},
              hover_data={'churn_rates': True, 'return_rates': True})
fig4.update_traces(hovertemplate="Gender: %{x}<br>Total: %{y:,.0f}<br>Churn Rate: %{customdata[0]}%<br>Return Rate: %{customdata[1]}%<extra></extra>")

for index, row in gender_summary.iterrows():
    fig4.add_annotation(x=row['Gender'], y=row['total_churned'],
                        text=f"{row['churn_rates']:.2f}%", showarrow=False, xanchor='right', yanchor='bottom',
                        font=dict(color='blue', size=12))
    fig4.add_annotation(x=row['Gender'], y=row['total_returned'],
                        text=f"{row['return_rates']:.2f}%", showarrow=False, xanchor='left', yanchor='bottom',
                        font=dict(color='red', size=12))

fig4.update_layout(title={'x': 0.5, 'y': 0.9}, width=600, height=500, xaxis_title=None, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
pio.write_html(fig4, file='templates/churn_returns_gender.html', auto_open=False)

# Age Group Analysis
age_bins = [0, 9, 19, 29, 39, 49, 59, 69, np.inf]
age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
df['Age Group'] = pd.cut(df['Customer Age'], bins=age_bins, labels=age_labels)
df['Age Group'] = pd.Categorical(df['Age Group'], categories=age_labels, ordered=True)

# Age Distribution
age_group_count = df['Age Group'].value_counts().sort_index().reset_index(name='Count')
age_group_count['Percentage'] = round(age_group_count['Count'] / age_group_count['Count'].sum() * 100, 2)
age_group_count['Percentage Text'] = age_group_count['Percentage'].astype(str) + '%'

fig1 = px.line(age_group_count, x='Age Group', y='Count', title='1. Age Distribution',
               markers=True, text='Percentage Text', labels={'Count': 'Number of Customers'})
fig1.update_traces(textposition="top center")
fig1.add_bar(x=age_group_count['Age Group'], y=age_group_count['Count'], text=age_group_count['Count'], textposition='inside', name='Count')
fig1.update_traces(textfont_color='white', selector=dict(type='bar'))
fig1.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, yaxis_title='Count', width=750, height=450, showlegend=False)
pio.write_html(fig1, file='templates/age_distribution.html', auto_open=False)

age_group_gender_count = df[['Gender', 'Age Group']].value_counts().reset_index(name='Count')

fig2 = px.bar(age_group_gender_count, x='Age Group', y='Count', color='Gender', barmode='group',
              color_discrete_map={'Female': '#FF6692', 'Male': '#3366CC'},
              category_orders={'Age Group': age_labels},
              title='2. Age Distribution by Gender', text_auto=True)
fig2.update_layout(title={'x': 0.5, 'y': 0.9}, width=750, height=450, xaxis_title=None, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
pio.write_html(fig2, file='templates/age_distribution_gender.html', auto_open=False)

# Product Analysis
df['Return Quantity'] = df.apply(lambda row: row['Quantity'] if row['Returns'] == 1 else 0, axis=1)
product_summary_table = df.groupby('Product Category').agg(
    Count=('Product Category', 'size'),
    Order_Quantity=('Quantity', 'sum'),
    Total_Revenue=('Total Price', 'sum'),
    Average_Revenue=('Total Price', 'mean'),
    Return_Count=('Return Quantity', 'sum')
).reset_index()

total_order_quantity = product_summary_table['Order_Quantity'].sum()
product_summary_table['Overall_Percentage'] = (product_summary_table['Order_Quantity'] / total_order_quantity) * 100
product_summary_table['Return_Rate'] = round((product_summary_table['Return_Count'] / product_summary_table['Order_Quantity']) * 100, 2)

category_order = ["Electronics", "Clothing", "Home", "Books"]

fig1 = px.bar(product_summary_table, x='Product Category', y=['Order_Quantity', 'Return_Count'],
              barmode='overlay', opacity=0.7, color='Product Category',
              color_discrete_sequence=px.colors.qualitative.T10,
              category_orders={'Product Category': category_order},
              title='Product Category Distribution vs. Returns', labels={'value': 'Return & Total Order Count'})

for i, row in product_summary_table.iterrows():
    fig1.add_annotation(x=row['Product Category'], y=row['Order_Quantity'] - 15000,
                        text=f"Total:{row['Order_Quantity'] / 1000:,.2f}K<br>({row['Overall_Percentage']:.2f}%)",
                        font=dict(color='black', size=12), showarrow=False)
    fig1.add_annotation(x=row['Product Category'], y=row['Return_Count'] - 15000,
                        text=f"Returned:{row['Return_Count'] / 1000:,.2f}K<br>({row['Return_Rate']:.2f}%)",
                        font=dict(color='white', size=12), showarrow=False)
fig1.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=700, height=500, showlegend=False)
pio.write_html(fig1, file='templates/product_category_distribution.html', auto_open=False)

total_revenue = df.groupby(['Product Category', 'Purchase Year'])['Total Price'].sum().reset_index()
total_revenue['Purchase Year'] = total_revenue['Purchase Year'].astype(str)

fig2 = px.bar(total_revenue, x='Product Category', y='Total Price', color='Purchase Year', barmode='group',
              color_discrete_sequence=px.colors.qualitative.Set2,
              category_orders={'Product Category': category_order},
              title='2. Sales Revenue by Product Category and Year', labels={'Total Price': 'Total Price ($)'})
fig2.update_traces(texttemplate='%{y:$.4s}', textposition='inside', textangle=90)
fig2.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=700, height=500)
pio.write_html(fig2, file='templates/sales_revenue_product_category_year.html', auto_open=False)

product_revenue_gender = df.groupby(['Product Category', 'Gender'])['Total Price'].sum().reset_index(name='Total Revenue')

fig3 = px.bar(product_revenue_gender, x='Product Category', y='Total Revenue', color='Gender',
              color_discrete_map={'Female': '#FF6692', 'Male': '#3366CC'}, barmode='group',
              title='3. Sales Revenue by Product Category & Gender',
              category_orders={'Product Category': category_order}, labels={'Total Revenue': 'Total Revenue ($)'})
fig3.update_traces(texttemplate='%{y:$.4s}', textposition='inside', textangle=90)
fig3.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=700, height=500, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
pio.write_html(fig3, file='templates/sales_revenue_product_category_gender.html', auto_open=False)

product_revenue_age = df.groupby(['Product Category', 'Age Group'])['Total Price'].sum().reset_index(name='Total Revenue')

fig4 = px.bar(product_revenue_age[product_revenue_age['Total Revenue'] > 0], x='Age Group', y='Total Revenue', color='Product Category', barmode='group',
              color_discrete_sequence=px.colors.qualitative.T10, category_orders={'Product Category': category_order},
              title='4. Product Sales Revenue by Category & Age Group', labels={'Total Revenue': 'Total Revenue ($)'})
fig4.update_traces(texttemplate='%{y:$.3s}', textangle=90)
fig4.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=800, height=500, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
pio.write_html(fig4, file='templates/sales_revenue_product_category_age.html', auto_open=False)

# Yearly Sales Revenue Summary
annual_sales_summary = df.groupby('Purchase Year')['Total Price'].agg(['min', 'max', 'mean', 'median', 'sum'])
print(annual_sales_summary)

# Monthly Sales Analysis
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['Purchase Month'] = pd.Categorical(df['Purchase Month'], categories=month_order, ordered=True)
monthly_sales = df.groupby(['Purchase Year', 'Purchase Month']).agg(
    Count=('Customer ID', 'size'), Total_Revenue=('Total Price', 'sum'), Avg_Revenue=('Total Price', 'mean')).reset_index()

fig1 = px.bar(monthly_sales, x='Purchase Month', y='Count', color='Purchase Year', facet_row='Purchase Year',
              title='1. Number of Orders per Month (by Year)', labels={'Count': 'Order Count', 'Purchase Year': 'Year'}, hover_data={'Count': ':,.0f'}, text_auto=True)
fig1.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=650, height=600, showlegend=False)
pio.write_html(fig1, file='templates/monthly_orders_per_year.html', auto_open=False)

fig2 = px.area(monthly_sales[monthly_sales['Total_Revenue'] != 0], x='Purchase Month', y='Total_Revenue', color='Purchase Year', facet_col='Purchase Year',
               color_discrete_sequence=px.colors.qualitative.Dark2,
               title='2. Total Monthly Sales Revenue (by Year)', labels={'Total_Revenue': 'Total Revenue ($)', 'Purchase Year': 'Year'}, hover_data={'Total_Revenue': ':$,.0f'}, markers=True)
fig2.update_yaxes(tickprefix="$")
fig2.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=800, height=350, showlegend=False)
pio.write_html(fig2, file='templates/total_monthly_sales_revenue.html', auto_open=False)

fig3 = px.line(monthly_sales, x='Purchase Month', y='Avg_Revenue', color='Purchase Year', facet_col='Purchase Year',
               color_discrete_sequence=px.colors.qualitative.Set2,
               title='3. Average Monthly Sales Revenue (by Year)', labels={'Avg_Revenue': 'Average Revenue ($)', 'Purchase Year': 'Year'}, hover_data={'Avg_Revenue': ':$,.0f'}, markers=True)
fig3.update_yaxes(tickprefix="$")
fig3.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=800, height=350, yaxis=dict(range=[700, 840]), showlegend=False)
pio.write_html(fig3, file='templates/average_monthly_sales_revenue.html', auto_open=False)

df_return_churn = df[['Customer ID', 'Purchase Date', 'Returns', 'Churn']]
df_return_churn['Purchase Date'] = df_return_churn['Purchase Date'].dt.strftime('%Y-%m')
rates = df_return_churn.groupby('Purchase Date').agg(Return_Rate=('Returns', 'mean'), Churn_Rate=('Churn', 'mean')).reset_index()
rates['Return_Rate'] *= 100
rates['Churn_Rate'] *= 100

fig4 = px.line(rates, x='Purchase Date', y=['Return_Rate', 'Churn_Rate'], title='4. Monthly Return and Churn Rates Over time', markers=True, hover_data={'variable': ':name'})
fig4.update_traces(hovertemplate="Type: %{customdata}<br>Purchase Date: %{x|%Y-%m}<br>Rate: %{y:.2f}%")

max_return_rate = rates.loc[rates['Return_Rate'].idxmax()]
min_return_rate = rates.loc[rates['Return_Rate'].idxmin()]
max_churn_rate = rates.loc[rates['Churn_Rate'].idxmax()]
min_churn_rate = rates.loc[rates['Churn_Rate'].idxmin()]

annotations = [
        {'x': max_churn_rate['Purchase Date'], 'y': max_churn_rate['Churn_Rate'] + 1,
     'text': f"highest: {max_churn_rate['Churn_Rate']:.2f}%", 'showarrow': False},
    {'x': min_churn_rate['Purchase Date'], 'y': min_churn_rate['Churn_Rate'] - 1,
     'text': f"lowest: {min_churn_rate['Churn_Rate']:.2f}%", 'showarrow': False}
]

for annotation in annotations:
    fig4.add_annotation(x=annotation['x'], y=annotation['y'], text=annotation['text'],
                        showarrow=annotation['showarrow'], font=dict(size=11))

fig4.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, yaxis_title='Rates (%)', width=750, height=450, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
pio.write_html(fig4, file='templates/monthly_return_churn_rates.html', auto_open=False)

# Customer Lifetime Value (CLV) Analysis
customer_df = df.groupby(['Customer ID', 'Customer Name']).agg(
    order_count=('Customer ID', 'size'),
    total_revenue=('Total Price', 'sum'),
    avg_purchase_value=('Total Price', 'mean'),
    return_num=('Returns', 'sum'),
    churned=('Churn', 'sum')
).reset_index()

customer_df['churned'] = customer_df['churned'].apply(lambda x: 1 if x >= 1 else 0)
customer_df['avg_purchase_freq'] = customer_df['order_count'] / len(customer_df)
customer_lifespans = df.groupby('Customer ID')['Purchase Date'].apply(lambda x: (x.max() - x.min()).days)
avg_lifespan_days = round(customer_lifespans.mean(), 1)
customer_df['customer_value'] = customer_df['avg_purchase_value'] * customer_df['avg_purchase_freq']
customer_df['CLV'] = customer_df['customer_value'] * avg_lifespan_days

clv_summary = customer_df['CLV'].describe()
high_cutoff = clv_summary['75%']
low_cutoff = clv_summary['25%']

def assign_segment(clv):
    if clv >= high_cutoff:
        return 'High'
    elif clv >= low_cutoff:
        return 'Medium'
    else:
        return 'Low'

customer_df['Segment'] = customer_df['CLV'].apply(assign_segment)

top_10_high_value_customers = customer_df[customer_df['Segment'] == 'High'].sort_values(by='CLV', ascending=False).head(10)

df = df.merge(customer_df[['Customer ID', 'Segment']], on='Customer ID', how='left')
df.rename(columns={'Segment': 'Segment Tier'}, inplace=True)

segment_summary = customer_df.groupby('Segment').agg(
    Total_Count=('Segment', 'size'),
    Total_Order_Count=('order_count', 'sum'),
    Total_Revenue=('total_revenue', 'sum'),
    Average_Revenue=('total_revenue', 'mean'),
    Total_Churned=('churned', 'sum'),
    Churn_Rates=('churned', 'mean'),
    Total_Return_Num=('return_num', 'sum')
).reset_index()
segment_summary['Return_Rates'] = segment_summary['Total_Return_Num'] / segment_summary['Total_Order_Count']

segment_order = ["Low", "Medium", "High"]

fig1 = px.pie(segment_summary, values='Total_Count', names='Segment', color='Segment',
              color_discrete_sequence=px.colors.qualitative.Dark24,
              category_orders={'Segment': segment_order}, title='1. Customer Segment Distribution')
fig1.update_traces(textposition='inside', textinfo='text', texttemplate='%{label}<br>%{value}<br>(%{percent})')
fig1.update_layout(title={'x': 0.5, 'y': 0.9}, width=400, height=400, showlegend=False)
pio.write_html(fig1, file='templates/customer_segment_distribution.html', auto_open=False)

sum_overall_revenue = segment_summary['Total_Revenue'].sum()
segment_summary['Revenue Percentage'] = (segment_summary['Total_Revenue'] / sum_overall_revenue) * 100

fig2 = px.bar(segment_summary, x='Segment', y='Total_Revenue', color='Segment',
              color_discrete_sequence=px.colors.qualitative.Dark24,
              title='2. Total Purchase Amount by Segment', labels={'Total_Revenue': 'Total Money Spent ($)'},
              category_orders={'Segment': segment_order},
              text=segment_summary.apply(lambda x: f"${x['Total_Revenue']:,}<br>({x['Revenue Percentage']:.2f}%)", axis=1),
              hover_data={'Total_Revenue': ':$,.0f'})
fig2.update_layout(title={'x': 0.5, 'y': 0.9}, width=550, height=450, showlegend=False)
pio.write_html(fig2, file='templates/total_purchase_amount_by_segment.html', auto_open=False)

segment_df = df[['Customer ID', 'Purchase Date', 'Total Price', 'Segment Tier']]
segment_df['Purchase Date'] = segment_df['Purchase Date'].dt.strftime('%Y-%m')
segment_monthly_sales = segment_df.groupby(['Purchase Date', 'Segment Tier'])['Total Price'].sum().reset_index(name='Sum')

fig3 = px.area(segment_monthly_sales, x='Purchase Date', y='Sum', color='Segment Tier',
               color_discrete_map={"Low": '#2E91E5', "Medium": '#E15F99', "High": '#1CA71C'},
               title='3. Monthly Sales Revenue Over Time by Segment', markers=True,
               symbol='Segment Tier', hover_data={'Segment Tier': ':name', 'Purchase Date': True, 'Sum': True})
fig3.update_traces(hovertemplate="Value: %{customdata}<br>Purchase Date: %{x|%Y-%m}<br>Sum: %{y:$,.3s}")
fig3.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, yaxis_title='Total Revenue ($)', width=750, height=500, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
pio.write_html(fig3, file='templates/monthly_sales_revenue_by_segment.html', auto_open=False)

segment_age_count = df.groupby(['Age Group', 'Segment Tier']).size().reset_index(name='Count')

fig4 = px.bar(segment_age_count, x='Age Group', y='Count', color='Segment Tier', barmode='group',
              color_discrete_sequence=px.colors.qualitative.Dark24,
              category_orders={'Segment Tier': segment_order, 'Age Group': age_labels},
              text_auto=True, title='4. Age Distribution by Segment')
fig4.update_layout(title={'x': 0.5, 'y': 0.9}, width=800, height=500)
pio.write_html(fig4, file='templates/age_distribution_by_segment.html', auto_open=False)

segment_summary['Churn_Rates'] *= 100
segment_summary['Return_Rates'] *= 100

fig5 = px.bar(segment_summary, x='Segment', y=['Total_Churned', 'Total_Return_Num'], barmode='group',
              category_orders={'Segment': segment_order},
              title='5. Customer Churn and Returns by Segment', text_auto='.3s',
              labels={'value': 'Number of Customers'}, hover_data={'Churn_Rates': True, 'Return_Rates': True})
fig5.update_traces(hovertemplate="Segment Tier: %{x}<br>Total: %{y:,.0f}<br>Churn Rate: %{customdata[0]}%<br>Return Rate: %{customdata[1]}%<extra></extra>")
fig5.update_traces(textfont={'size': 11}, textposition='inside')

for index, row in segment_summary.iterrows():
    fig5.add_annotation(x=row['Segment'], y=row['Total_Churned'],
                        text=f"{row['Churn_Rates']:.2f}%", showarrow=False, xanchor='right', yanchor='bottom',
                        font=dict(color='blue', size=11))
    fig5.add_annotation(x=row['Segment'], y=row['Total_Return_Num'],
                        text=f"{row['Return_Rates']:.2f}%", showarrow=False, xanchor='left', yanchor='bottom',
                        font=dict(color='red', size=11))

fig5.update_layout(title={'x': 0.5, 'y': 0.9}, width=600, height=500, xaxis_title=None, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
pio.write_html(fig5, file='templates/customer_churn_and_returns_by_segment.html', auto_open=False)

# Payment Method Analysis
payment_count_returns = df.groupby('Payment Method').agg(
    total_count=('Customer ID', 'size'),
    total_return=('Returns', 'sum'),
    return_rates=('Returns', 'mean')
).reset_index()
payment_count_returns['percentage'] = payment_count_returns['total_count'] / payment_count_returns['total_count'].sum() * 100
payment_count_returns['return_rates'] *= 100

payment_method_order = ['Cash', 'Credit Card', 'PayPal']

fig1 = px.bar(payment_count_returns, x='Payment Method', y=['total_count', 'total_return'], barmode='overlay', opacity=0.7,
              color='Payment Method', color_discrete_sequence=px.colors.qualitative.Prism,
              category_orders={'Payment Method': payment_method_order},
              title='1. Payment Method Distribution vs. Returns', labels={'value': 'Return & Total Order Count'})

for i, row in payment_count_returns.iterrows():
    fig1.add_annotation(x=row['Payment Method'], y=row['total_count'] - 5000,
                        text=f"Total:{row['total_count'] / 1000:,.2f}K<br>({row['percentage']:.2f}%)",
                        font=dict(color='black', size=12), showarrow=False)
    fig1.add_annotation(x=row['Payment Method'], y=row['total_return'] - 5000,
                        text=f"Returned:{row['total_return'] / 1000:,.2f}K<br>({row['return_rates']:.2f}%)",
                        font=dict(color='white', size=12), showarrow=False)
fig1.update_traces(hovertemplate="Payment Method: %{x}<br>Count: %{y:,.0f}<extra></extra>")
fig1.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title=None, width=600, height=550, showlegend=False)
pio.write_html(fig1, file='templates/payment_method_distribution.html', auto_open=False)

payment_gender_age = df.groupby(['Payment Method', 'Gender', 'Age Group']).agg(Count=('Customer ID', 'size')).reset_index()

fig2 = px.bar(payment_gender_age, x='Age Group', y='Count', barmode='group', facet_col='Payment Method',
              color='Gender', color_discrete_map={'Female': '#FF6692', 'Male': '#3366CC'},
              category_orders={'Payment Method': payment_method_order},
              title='2. Payment Method Distribution by Gender and Age', labels={'Count': 'Number of Customers'}, text_auto=True)
fig2.update_traces(width=0.5)
fig2.update_layout(title={'x': 0.5, 'y': 0.95}, bargap=0.2, xaxis_title=None, xaxis2_title=None, xaxis3_title=None, width=900, height=400, legend_title=None, legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="center", x=0.5))
pio.write_html(fig2, file='templates/payment_method_gender_age.html', auto_open=False)

payment_segment = df.groupby(['Payment Method', 'Segment Tier']).agg(Count=('Customer ID', 'size')).reset_index()
payment_segment['Percentage'] = payment_segment['Count'] / payment_segment['Count'].sum()

fig3 = px.bar(payment_segment, x='Payment Method', y='Percentage', barmode='group',
              color='Segment Tier', color_discrete_sequence=px.colors.qualitative.Dark24,
              category_orders={'Payment Method': payment_method_order, 'Segment Tier': segment_order},
              title='3. Payment Method Distribution by Segment', labels={'Percentage': 'Distribution (%)'}, text_auto='.1%')
fig3.update_layout(title={'x': 0.5, 'y': 0.9}, bargap=0.1, xaxis_title=None, width=600, height=450)
pio.write_html(fig3, file='templates/payment_method_segment.html', auto_open=False)

# Correlation Analysis
df_corr = df[['Product Price', 'Quantity', 'Total Price', 'Customer Age', 'Returns', 'Churn']].corr(method='pearson').round(3)
fig = px.imshow(df_corr, color_continuous_scale='blugrn', text_auto=True, aspect="auto")
fig.update_layout(title={'x': 0.5, 'y': 0.9}, width=800, height=450)
pio.write_html(fig, file='templates/correlation_analysis.html', auto_open=False)

fig = px.scatter(rates, x='Return_Rate', y='Churn_Rate', trendline="ols", trendline_scope="overall")
rates_corr = rates['Return_Rate'].corr(rates['Churn_Rate'])
fig.add_annotation(x=41.5, y=20.15, text=f'R = {rates_corr:.3f}', showarrow=False, font=dict(size=13, color="red"))
fig.update_layout(title={'x': 0.5, 'y': 0.9}, xaxis_title='Return Rate (%)', yaxis_title='Churn Rate (%)', width=700, height=450)
pio.write_html(fig, file='templates/return_churn_correlation.html', auto_open=False)

print("All visualizations have been saved successfully.")


