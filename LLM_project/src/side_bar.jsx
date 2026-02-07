import React, {useState} from "react";

function SideBar(){

    const [pdfs, setPdfs] = useState([]);
    const [selectedPdf, setSelectedPdf] = useState(null);
    const [selectedChat, setSelectedChat] = useState(null);
    const [chats, setChats] = useState({});

    const createChat = (pdf_id) => {

            
    }

    return(<>
        <button></button>
        <button className="all-docs">All</button>
    </>
    );
}

export default SideBar;