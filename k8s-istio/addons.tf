resource "aws_eks_addon" "ebs_csi" {
  cluster_name  = aws_eks_cluster.eks_cluster.name
  addon_name    = "aws-ebs-csi-driver"
  addon_version = "v1.29.1-eksbuild.1"
  depends_on = [ aws_eks_node_group.private-nodes]
}
