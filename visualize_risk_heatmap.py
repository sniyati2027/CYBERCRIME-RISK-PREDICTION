import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_heatmap():
    # 1. Load Data
    try:
        df = pd.read_csv('city_risk_stats.csv')
    except FileNotFoundError:
        print("Error: city_risk_stats.csv not found. Run generate_risk_intelligence.py first.")
        return

    # 2. Define Coordinates (Approximate Lat/Lon for major Indian cities)
    # withdrawal_city names must match specific keys.
    city_coords = {
        'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
        'Delhi': {'lat': 28.7041, 'lon': 77.1025},
        'Bengaluru': {'lat': 12.9716, 'lon': 77.5946},
        'Lucknow': {'lat': 26.8467, 'lon': 80.9462},
        'Jaipur': {'lat': 26.9124, 'lon': 75.7873},
        'Indore': {'lat': 22.7196, 'lon': 75.8577},
        'Pune': {'lat': 18.5204, 'lon': 73.8567},
        'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
        'Chennai': {'lat': 13.0827, 'lon': 80.2707},
        'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
        'Surat': {'lat': 21.1702, 'lon': 72.8311}
    }

    # Map coordinates
    df['lat'] = df['withdrawal_city'].map(lambda x: city_coords.get(x, {}).get('lat'))
    df['lon'] = df['withdrawal_city'].map(lambda x: city_coords.get(x, {}).get('lon'))

    # Drop cities without coordinates (if any new ones appear)
    df = df.dropna(subset=['lat', 'lon'])

    # 3. Plotting
    plt.figure(figsize=(10, 10))
    
    # Create the scatter plot
    # Size based on calculated risk or total complaints? Risk Score makes more sense for "Risk Heatmap"
    # But prompt asks "Color intensity should represent Avg_Risk_Score"
    # Let's verify: "Color intensity should represent Avg_Risk_Score"
    
    # We can use size for Totals and Color for Risk.
    
    scatter = plt.scatter(
        x=df['lon'], 
        y=df['lat'], 
        c=df['Avg_Risk_Score'], 
        s=df['Total_Complaints'] * 0.5, # Scale size
        cmap='Reds',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )

    # 4. Annotations
    for i, row in df.iterrows():
        # Label High Priority Zones
        if row['Priority_Level'] == 'High Priority Monitoring Zone':
            label_text = f"{row['withdrawal_city']}\n(High Risk)"
            font_weight = 'bold'
        else:
            label_text = row['withdrawal_city']
            font_weight = 'normal'
            
        plt.text(
            row['lon'] + 0.5, 
            row['lat'] + 0.2, 
            label_text, 
            fontsize=9, 
            weight=font_weight
        )

    # 5. Styling
    plt.title('Geographic Risk Heatmap: Withdrawal Hotspots', fontsize=16, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Risk Score (0-1)', rotation=270, labelpad=15)
    
    # Add specific legend for High Priority
    # We can add a text box
    textstr = '\n'.join((
        r'$\bf{High\ Priority\ Zones}$',
        'Mumbai, Kolkata, Delhi',
        'Bengaluru, Lucknow'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save
    plt.savefig('risk_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap saved: risk_heatmap.png")

if __name__ == "__main__":
    visualize_heatmap()
