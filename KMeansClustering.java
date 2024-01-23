import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class KMeansClustering {

    public static class KMeansMapper extends Mapper<LongWritable, Text, VectorWritable, NullWritable> {

        private Vector vector;
        private Map<Float, Vector> clusterCenters;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            vector = new org.apache.mahout.math.DenseVector(5); // Use DenseVector for efficiency
            // Load the cluster centers from the input path
            Path centersPath = new Path(context.getConfiguration().get("centersPath"));
            List<VectorWritable> centers = new SequenceFileIterable<VectorWritable, NullWritable>(centersPath, VectorWritable.class, NullWritable.class).collect();
            for (VectorWritable center : centers) {
                clusterCenters.put(center.get().get(0), center.get());
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] values = value.toString().split(",");
            // Ensure correct indexing for O3, CO, NO2, SO2, PM2.5 based on data format
            vector.set(0, Float.parseFloat(values[1]));
            vector.set(1, Float.parseFloat(values[2]));
            vector.set(2, Float.parseFloat(values[3]));
            vector.set(3, Float.parseFloat(values[4]));
            vector.set(4, Float.parseFloat(values[5]));

            // Find the closest cluster center
            Float closestClusterId = Float.NEGATIVE_INFINITY;
            Vector closestClusterCenter = null;
            for (Map.Entry<Float, Vector> entry : clusterCenters.entrySet()) {
                Float distance = euclideanDistance(vector, entry.getValue());
                if (distance < closestClusterId) {
                    closestClusterId = distance;
                    closestClusterCenter = entry.getValue();
                }
            }

            // Write the point to the output with the cluster ID as the key
            context.write(new VectorWritable(vector), new FloatWritable(closestClusterId));
        }
