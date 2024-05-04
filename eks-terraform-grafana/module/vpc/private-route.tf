resource "aws_route_table" "private-route-table" {
  vpc_id = aws_vpc.vpc.id
  
  tags = {
    Name  = "Private-route"
    from  = "terraform"
    owner = "manoj"
  }
}

resource "aws_route" "nat_route" {
  count = var.nat_server_status ? 1 : 0  
  route_table_id         = aws_route_table.private-route-table.id
  destination_cidr_block = "0.0.0.0/0"
  network_interface_id   = aws_instance.nat_server[count.index].primary_network_interface_id
}

resource "aws_route_table_association" "association_private" {
  route_table_id = aws_route_table.private-route-table.id
  count          = length(var.private-cidr)  
  subnet_id      = aws_subnet.private-subnet[count.index].id
}
