resource "aws_eks_addon" "ebs_csi" {
  cluster_name = aws_eks_cluster.eks_cluster.name
  addon_name   = "ebs-csi-driver"
}
