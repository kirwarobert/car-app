import streamlit as st
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

st.title("NicheReach Ad Server Simulation")
st.sidebar.title("Simulation Controls")

# Add input for number of page views in the sidebar
num_page_views = st.sidebar.number_input(
    "Number of Page Views to Simulate",
    min_value=100,
    max_value=100000,
    value=5000,
    step=100
)

# Add a button to run the simulation in the sidebar
run_simulation_button = st.sidebar.button("Run Simulation")


# ======================
# 1. DEFINE DATA STRUCTURES (Our "Database")
# ======================

class Ad:
    """A class to represent a single advertisement."""
    def __init__(self, ad_id: int, advertiser_id: int, campaign_id: int, max_cpc_bid: float, headline: str, target_publisher_ids: List[int]):
        self.ad_id = ad_id
        self.advertiser_id = advertiser_id
        self.campaign_id = campaign_id
        self.max_cpc_bid = max_cpc_bid  # The max amount the advertiser will pay for a click
        self.headline = headline
        self.target_publisher_ids = target_publisher_ids # Which publisher sites this ad can appear on
        self.total_clicks = 0
        self.total_impressions = 0

    @property
    def ctr(self):
        """Calculate Click-Through Rate."""
        if self.total_impressions == 0:
            return 0.0
        return self.total_clicks / self.total_impressions

    def __repr__(self):
        return f"Ad(ID: {self.ad_id}, Bid: ${self.max_cpc_bid:.2f}, CTR: {self.ctr:.2%}) - '{self.headline}'"

class Publisher:
    """A class to represent a publisher site where ads are shown."""
    def __init__(self, publisher_id: int, name: str, category: str, traffic_weight: float = 1.0):
        self.publisher_id = publisher_id
        self.name = name
        self.category = category
        self.traffic_weight = traffic_weight # Simulates high vs. low traffic sites

class AdServer:
    """The core simulation engine that manages ads, publishers, and runs auctions."""
    def __init__(self):
        self.ads: Dict[int, Ad] = {}
        self.publishers: Dict[int, Publisher] = {}
        self.auction_history = []

    def register_ad(self, ad: Ad):
        """Add an ad to the server's inventory."""
        self.ads[ad.ad_id] = ad

    def register_publisher(self, publisher: Publisher):
        """Add a publisher to the network."""
        self.publishers[publisher.publisher_id] = publisher

    def run_auction(self, publisher_id: int) -> Optional[Ad]:
        """Simulate an ad auction for a given publisher.
        Returns the winning ad or None if no ad is found."""
        # 1. Find eligible ads that target this publisher
        eligible_ads = [ad for ad in self.ads.values() if publisher_id in ad.target_publisher_ids]

        if not eligible_ads:
            return None

        # 2. Simulate a "Quality Score" (e.g., based on historical CTR)
        # In reality, this is a complex metric. We'll use a simple version.
        for ad in eligible_ads:
            # Base quality score is the CTR. Add a small random factor for realism.
            quality_score = max(0.01, ad.ctr) + random.uniform(0, 0.05)
            # The effective bid for ranking is bid * quality_score
            ad.effective_rank_score = ad.max_cpc_bid * quality_score

        # 3. Sort by effective rank score (highest wins the auction)
        eligible_ads.sort(key=lambda x: x.effective_rank_score, reverse=True)
        winning_ad = eligible_ads[0]

        # 4. The price the winner pays is determined by the second-price auction.
        # They pay just enough to beat the second-place ad.
        if len(eligible_ads) > 1:
            second_highest_score = eligible_ads[1].effective_rank_score
            winning_ad_quality = max(0.01, winning_ad.ctr) # Recalculate winning ad's quality
            # Price = (Second Place Score / Winner's Quality Score) + $0.01
            winning_ad_price = (second_highest_score / winning_ad_quality) + 0.01
        else:
            winning_ad_price = winning_ad.max_cpc_bid # If no competition, pay your max bid

        winning_ad.price_paid = winning_ad_price

        # 5. Record the impression for the winning ad
        winning_ad.total_impressions += 1

        # 6. Simulate a user click based on the ad's historical CTR + randomness
        click_chance = winning_ad.ctr if winning_ad.ctr > 0 else 0.02 # Default CTR for new ads
        click_chance += random.uniform(-0.01, 0.01) # Add noise
        did_click = random.random() < click_chance

        if did_click:
            winning_ad.total_clicks += 1
            # In a real system, we'd deduct `winning_ad.price_paid` from the advertiser's balance here.

        # Record the auction result for analysis
        self.auction_history.append({
            'publisher_id': publisher_id,
            'winning_ad_id': winning_ad.ad_id,
            'winning_bid': winning_ad.max_cpc_bid,
            'price_paid': winning_ad_price,
            'did_click': did_click
        })

        return winning_ad

# ======================
# 2. SETUP THE SIMULATION
# ======================

# Create our ad server
server = AdServer()

# Create some publishers
publishers_data = [
    (1, "EcoWarrior News", "eco_news", 1.5),
    (2, "Sustainable Style Blog", "sustainable_fashion", 1.2),
    (3, "Green Tech Today", "green_tech", 0.8),
]
for pid, name, cat, weight in publishers_data:
    server.register_publisher(Publisher(pid, name, cat, weight))

# Create some ads from different advertisers
ads_data = [
    # (ad_id, adv_id, camp_id, max_cpc_bid, headline, target_pubs)
    (101, 1, 1001, 1.20, "Organic Cotton T-Shirts - 50% Off Today!", [1, 2]),
    (102, 1, 1001, 0.85, "Compostable Phone Cases - Free Shipping!", [1, 3]),
    (201, 2, 2001, 2.50, "The Best Reusable Coffee Cups - Buy Now!", [1, 2, 3]),
    (202, 2, 2001, 1.75, "Solar Powered Chargers | Shop the Collection", [3]),
    (301, 3, 3001, 0.60, "DIY Home Composting Kit - On Sale Now", [1]),
    (302, 3, 3001, 0.90, "Bamboo Toothbrushes - 5 Pack for $15", [1, 2]),
]

for ad in ads_data:
    server.register_ad(Ad(*ad))


# ======================
# 3. RUN THE SIMULATION (Conditional based on button click)
# ======================

if run_simulation_button:
    st.write(f"🎯 Starting Simulation: {num_page_views} Page Views...")

    # Simulate page views across our publisher network
    for _ in range(num_page_views):
        # Randomly select a publisher, weighted by their traffic
        pub_id = random.choices(
            population=list(server.publishers.keys()),
            weights=[p.traffic_weight for p in server.publishers.values()],
            k=1
        )[0]
        # Run an auction for this page view
        server.run_auction(pub_id)

    st.write("Simulation complete!")

    # ======================
    # 4. ANALYZE THE RESULTS
    # ======================

    st.write("\n📊 Simulation Results Summary")
    st.write("=============================")

    # Create a DataFrame from the auction history
    history_df = pd.DataFrame(server.auction_history)

    # Calculate key metrics for each ad
    results = []
    for ad in server.ads.values():
        ad_history = history_df[history_df['winning_ad_id'] == ad.ad_id]
        impressions = len(ad_history)
        clicks = ad_history['did_click'].sum()
        total_spend = ad_history['price_paid'].sum()

        results.append({
            'Ad ID': ad.ad_id,
            'Headline': ad.headline[:30] + "...",
            'Max CPC Bid': f"${ad.max_cpc_bid:.2f}",
            'Impressions': impressions,
            'Clicks': clicks,
            'CTR': f"{clicks/impressions:.2%}" if impressions > 0 else "N/A",
            'Total Spend': f"${total_spend:.2f}",
            'Avg. CPC': f"${total_spend/clicks:.2f}" if clicks > 0 else "N/A"
        })

    results_df = pd.DataFrame(results).sort_values('Impressions', ascending=False)
    st.dataframe(results_df) # Display the results DataFrame

    # Analyze overall platform performance
    total_revenue = history_df['price_paid'].sum()
    total_clicks = history_df['did_click'].sum()
    # Avoid division by zero for fill_rate if num_page_views is 0
    fill_rate = (len(history_df) / num_page_views) * 100 if num_page_views > 0 else 0

    st.write(f"\n📈 Platform Performance:")
    st.write(f"   Total Revenue Simulated: ${total_revenue:.2f}")
    st.write(f"   Total Clicks Generated: {total_clicks}")
    st.write(f"   Average CPC: ${total_revenue/total_clicks:.2f}" if total_clicks > 0 else "No clicks")
    st.write(f"   Auction Fill Rate: {fill_rate:.1f}%")

    # ======================
    # 5. VISUALIZE THE RESULTS
    # ======================

    st.write("\n📊 Visualizations")

    # Plot the performance of each ad by spend and clicks
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Data for plotting - filter out ads with 0 impressions for plotting
    plottable_results_df = results_df[results_df['Impressions'] > 0]
    ad_ids = [f"Ad {r['Ad ID']}" for _, r in plottable_results_df.iterrows()]
    spend = [float(r['Total Spend'].replace('$', '')) for _, r in plottable_results_df.iterrows()]
    clicks = [r['Clicks'] for _, r in plottable_results_df.iterrows()]

    # Plot 1: Total Spend per Ad
    if ad_ids: # Check if there's data to plot
        ax1.bar(ad_ids, spend, color='skyblue')
        ax1.set_title('Total Ad Spend (Platform Revenue)')
        ax1.set_ylabel('Dollars ($)')
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.set_title('Total Ad Spend (Platform Revenue)')
        ax1.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)


    # Plot 2: Clicks per Ad
    if ad_ids: # Check if there's data to plot
        ax2.bar(ad_ids, clicks, color='lightgreen')
        ax2.set_title('Total Clicks Generated')
        ax2.set_ylabel('Number of Clicks')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.set_title('Total Clicks Generated')
        ax2.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)


    plt.tight_layout()
    st.pyplot(fig) # Display the figure in Streamlit

    # Plot the auction price distribution
    if not history_df.empty:
        plt.figure(figsize=(10, 5))
        plt.hist(history_df['price_paid'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        plt.title('Distribution of Winning Auction Prices (Cost Per Click)')
        plt.xlabel('Price Paid ($)')
        plt.ylabel('Frequency')
        plt.axvline(history_df['price_paid'].mean(), color='red', linestyle='dashed', linewidth=1, label=f'Mean: ${history_df["price_paid"].mean():.2f}')
        plt.legend()
        st.pyplot(plt) # Display the figure in Streamlit
    else:
        st.write("No auction history to plot price distribution.")
