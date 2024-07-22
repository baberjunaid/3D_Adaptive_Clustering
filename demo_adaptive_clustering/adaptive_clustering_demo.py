import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def load_pcd(file_path):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(file_path)
    # Convert the point cloud to a numpy array
    points_np = np.asarray(pcd.points)
    
    return points_np

class AdaptiveClustering:
    def __init__(self, z_axis_min=-0.1, z_axis_max=2.0, cluster_size_min=3, cluster_size_max=2200000):
        self.z_axis_min = z_axis_min
        self.z_axis_max = z_axis_max
        self.cluster_size_min = cluster_size_min
        self.cluster_size_max = cluster_size_max
        self.region_max = 10
        self.regions = [2, 3, 3, 3, 3, 3, 3, 2, 3, 3]  # Simplified regions for VLP-16

    def filter_ground_and_ceiling(self, points):
        return points[(points[:, 2] >= self.z_axis_min) & (points[:, 2] <= self.z_axis_max)]

    def divide_to_regions(self, points):
        regions = [[] for _ in range(self.region_max)]
        ranges = np.cumsum([0] + self.regions)
        
        distances = np.linalg.norm(points[:, :2], axis=1)
        for i in range(self.region_max):
            mask = (distances > ranges[i]) & (distances <= ranges[i+1])
            regions[i] = points[mask]
        
        return regions

    def adaptive_clustering(self, points):
        filtered_points = self.filter_ground_and_ceiling(points)
        regions = self.divide_to_regions(filtered_points)
        
        clusters = []
        for i, region in enumerate(regions):
            if len(region) < self.cluster_size_min:
                continue
            
            eps = 0.08 * (i + 1)  # Increasing epsilon for farther regions
            dbscan = DBSCAN(eps=eps, min_samples=self.cluster_size_min)
            labels = dbscan.fit_predict(region)
            
            for label in set(labels):
                if label == -1:  # Noise points
                    continue
                cluster = region[labels == label]
                if self.cluster_size_min <= len(cluster) <= self.cluster_size_max:
                    clusters.append(cluster)
        
        return clusters
    def visualize_original(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for original points
        o3d.visualization.draw_geometries([pcd])

    def visualize_clusters(self, points, clusters):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for original points

        cluster_pcds = []
        bounding_boxes = []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        for i, cluster in enumerate(clusters):
            if len(cluster) < 4:
                continue  # Skip clusters with less than 4 points

            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
            color = colors[i % len(colors)]
            cluster_pcd.paint_uniform_color(color)
            cluster_pcds.append(cluster_pcd)

            # Create oriented bounding box
            try:
                obb = cluster_pcd.get_oriented_bounding_box()
                obb.color = color
                bounding_boxes.append(obb)
            except RuntimeError as e:
                print(f"Could not create bounding box for cluster {i}: {e}")

        o3d.visualization.draw_geometries([pcd] + cluster_pcds + bounding_boxes)
    def visualize_combined(self, points, clusters):
        # Create a point cloud object for the original points
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(points)
        original_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray color for original points

        cluster_pcds = []
        bounding_boxes = []
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        
        for i, cluster in enumerate(clusters):
            if len(cluster) < 4:
                continue  # Skip clusters with less than 4 points

            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
            color = colors[i % len(colors)]
            cluster_pcd.paint_uniform_color(color)
            cluster_pcds.append(cluster_pcd)

            # Create oriented bounding box
            try:
                obb = cluster_pcd.get_oriented_bounding_box()
                obb.color = color
                bounding_boxes.append(obb)
            except RuntimeError as e:
                print(f"Could not create bounding box for cluster {i}: {e}")

        # Visualize everything together
        o3d.visualization.draw_geometries([original_pcd] + cluster_pcds + bounding_boxes)


def main():
    points = load_pcd('pcd_files/sample_1.pcd')
 
    ac = AdaptiveClustering()
    clusters = ac.adaptive_clustering(points)
    # This is to draw the original given point cloud
 
#     print("Visualizing original point cloud. Close the window to continue.")
#     ac.visualize_original(points)
    
    # printing the clusters size
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i} has {len(cluster)} points")

   # draw points and clusters
#     print("Visualizing clustered results. Close the window to exit.")
#     ac.visualize_clusters(points, clusters)
    
    # draw original point cloud + the clusters
    print("Visualizing original point cloud and clustered results. Close the window to exit.")
    ac.visualize_combined(points, clusters)

if __name__ == "__main__":
    main()