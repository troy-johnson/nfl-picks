import type { Metadata } from "next";
import "./styles.css";
export const metadata: Metadata={title:"NFL Picks",description:"Weekly NFL pick'em projections built from market and efficiency data."};
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>){return <html lang="en"><body>{children}</body></html>}
